import time

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from config import config
from utils.rotations import axis_rotation, axis_angle_rotation, rotation_to_axis_angle


def get_rotation_matrix(body_id):
    _, body_quat = p.getBasePositionAndOrientation(body_id)
    return np.array(p.getMatrixFromQuaternion(body_quat)).reshape(3, 3)


def set_orientation(body_id, R):
    body_pos, _ = p.getBasePositionAndOrientation(body_id)
    p.resetBasePositionAndOrientation(body_id, body_pos, Rotation.from_matrix(R).as_quat())


def show_label(text, body_id, replace_id=None, color=(0, 0, 0), size=1.5, height=1.3):
    body_pos, _ = p.getBasePositionAndOrientation(body_id)
    text_pos = [body_pos[0], body_pos[1], body_pos[2] + height]
    if replace_id is None:
        return p.addUserDebugText(text, text_pos, textColorRGB=color, textSize=size)
    return p.addUserDebugText(text, text_pos, textColorRGB=color, textSize=size, replaceItemUniqueId=replace_id)


def step_label(body_name, rotations, step, extrinsic=False):
    if step >= len(rotations):
        return f"{body_name}: done"
    axis, angle_rad = rotations[step]
    frame = "world" if extrinsic else "own"
    return f"{body_name} step {step + 1}: {np.degrees(angle_rad):.0f} deg around {frame} {axis}"


def wait_frames(frames):
    for _ in range(frames):
        step_start_time = time.time()
        p.stepSimulation()
        elapsed_time = time.time() - step_start_time
        time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))


def animate_rotations(bodies, extrinsic=False, after_step=None):
    # extrinsic=False: around own axes  R_{k-1} @ R_axis(t * angle)
    # extrinsic=True:  around world axes R_axis(t * angle) @ R_{k-1}
    # after_step(step): optional callback, called when step `step` (0-based) is finished
    pause_frames = config.Config.FPS
    steps_count = max(len(rotations) for _, _, rotations, _ in bodies)
    label_ids = {body_id: None for body_id, _, _, _ in bodies}

    for step in range(steps_count):
        for body_id, body_name, rotations, _ in bodies:
            label_ids[body_id] = show_label(step_label(body_name, rotations, step, extrinsic), body_id, label_ids[body_id])

        for i in range(config.Config.OBJECT_ROTATION_STEPS):
            step_start_time = time.time()
            p.stepSimulation()
            elapsed_time = time.time() - step_start_time

            t = i / (config.Config.OBJECT_ROTATION_STEPS - 1)
            for body_id, _, rotations, steps_World in bodies:
                if step < len(rotations):
                    axis, angle_rad = rotations[step]
                    R_axis = axis_rotation(axis, t * angle_rad)
                    set_orientation(body_id, R_axis @ steps_World[step] if extrinsic else steps_World[step] @ R_axis)

            time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))

        if after_step is not None:
            after_step(step)
        wait_frames(pause_frames)

    for body_id, body_name, _, steps_World in bodies:
        set_orientation(body_id, steps_World[-1])
        label_ids[body_id] = show_label(f"{body_name}: done", body_id, label_ids[body_id])


def animate_axis_angle_rotation(body_id, R_start, axis_World, angle_rad, label):
    # R(t) = Rot(axis_World, t * angle) @ R_start - rotation around a fixed WORLD axis through the body position
    body_pos, _ = p.getBasePositionAndOrientation(body_id)
    axis_World = np.asarray(axis_World) / np.linalg.norm(axis_World)
    p.addUserDebugLine(np.array(body_pos) - 1.2 * axis_World, np.array(body_pos) + 1.2 * axis_World,
                       lineColorRGB=[1, 0, 1], lineWidth=3)
    label_id = show_label(label, body_id)

    for i in range(config.Config.OBJECT_ROTATION_STEPS):
        step_start_time = time.time()
        p.stepSimulation()
        elapsed_time = time.time() - step_start_time

        t = i / (config.Config.OBJECT_ROTATION_STEPS - 1)
        set_orientation(body_id, axis_angle_rotation(axis_World, t * angle_rad) @ R_start)

        time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))

    return label_id


def animate_applied_rotations(bodies):
    """
    bodies: list of (body_id, body_name, applied, R_start)
      applied - list of (label, R_apply), each R_apply is applied in the WORLD frame: R_new = R_apply @ R_old.
    All bodies do their k-th rotation at the same time, as one turn around the axis of R_apply (axis-angle).
    Returns the final orientation of each body.
    """
    pause_frames = config.Config.FPS
    steps_count = max(len(applied) for _, _, applied, _ in bodies)
    R_current = {body_id: R_start for body_id, _, _, R_start in bodies}
    label_ids = {body_id: None for body_id, _, _, _ in bodies}

    for body_id, _, _, R_start in bodies:
        set_orientation(body_id, R_start)

    for step in range(steps_count):
        line_ids = []
        motions = []
        for body_id, body_name, applied, _ in bodies:
            if step >= len(applied):
                continue
            label, R_apply = applied[step]
            axis_World, angle_rad = rotation_to_axis_angle(R_apply)
            label_ids[body_id] = show_label(f"{body_name} step {step + 1}: apply {label} "
                                            f"({np.degrees(angle_rad):.1f} deg)", body_id, label_ids[body_id])
            body_pos, _ = p.getBasePositionAndOrientation(body_id)
            line_ids.append(p.addUserDebugLine(np.array(body_pos) - 1.2 * axis_World,
                                               np.array(body_pos) + 1.2 * axis_World,
                                               lineColorRGB=[1, 0, 1], lineWidth=3))
            motions.append((body_id, axis_World, angle_rad, R_current[body_id]))

        for i in range(config.Config.OBJECT_ROTATION_STEPS):
            step_start_time = time.time()
            p.stepSimulation()
            elapsed_time = time.time() - step_start_time

            t = i / (config.Config.OBJECT_ROTATION_STEPS - 1)
            for body_id, axis_World, angle_rad, R_old in motions:
                set_orientation(body_id, axis_angle_rotation(axis_World, t * angle_rad) @ R_old)

            time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))

        for body_id, body_name, applied, _ in bodies:
            if step < len(applied):
                R_current[body_id] = applied[step][1] @ R_current[body_id]
                set_orientation(body_id, R_current[body_id])

        wait_frames(pause_frames)
        for line_id in line_ids:
            p.removeUserDebugItem(line_id)

    for body_id, body_name, _, _ in bodies:
        label_ids[body_id] = show_label(f"{body_name}: done", body_id, label_ids[body_id])

    return [R_current[body_id] for body_id, _, _, _ in bodies]


_arrow_body_ids = []


def draw_arrow(start, direction, color, length=2.0, radius=0.04, tip_radius=0.09):
    # A 3D arrow (cylinder + ball on the tip) - visible on any GPU, unlike debug lines whose width is often ignored
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])  # pybullet cylinder is along its local Z
    axis = np.cross(z, direction)
    if np.linalg.norm(axis) < 1e-9:
        R = np.eye(3) if direction[2] > 0 else axis_angle_rotation([1, 0, 0], np.pi)
    else:
        R = axis_angle_rotation(axis, np.arccos(np.clip(z @ direction, -1.0, 1.0)))
    quat = Rotation.from_matrix(R).as_quat()
    rgba = list(color) + [1]

    shaft = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
    tip = p.createVisualShape(p.GEOM_SPHERE, radius=tip_radius, rgbaColor=rgba)
    start = np.asarray(start, dtype=float)
    _arrow_body_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=shaft,
                                             basePosition=start + direction * length / 2, baseOrientation=quat))
    _arrow_body_ids.append(p.createMultiBody(baseMass=0, baseVisualShapeIndex=tip,
                                             basePosition=start + direction * length))
    return start + direction * length


def remove_arrows():
    for body_id in _arrow_body_ids:
        p.removeBody(body_id)
    _arrow_body_ids.clear()


def set_position(body_id, position):
    _, body_quat = p.getBasePositionAndOrientation(body_id)
    p.resetBasePositionAndOrientation(body_id, position, body_quat)


def animate_translations(bodies):
    pause_frames = config.Config.FPS
    steps_count = max(len(moves) for _, _, moves in bodies)
    label_ids = {body_id: None for body_id, _, _ in bodies}
    colors = [[1, 0.5, 0], [0.8, 0, 0.8], [0, 0.6, 0.6]]

    for step in range(steps_count):
        motions = []
        for body_id, body_name, moves in bodies:
            if step >= len(moves):
                continue
            label, displacement_World = moves[step]
            start = np.array(p.getBasePositionAndOrientation(body_id)[0])
            label_ids[body_id] = show_label(f"{body_name} step {step + 1}: {label}", body_id, label_ids[body_id])
            motions.append((body_id, start, np.asarray(displacement_World, dtype=float)))

        for i in range(config.Config.OBJECT_ROTATION_STEPS):
            step_start_time = time.time()
            p.stepSimulation()
            elapsed_time = time.time() - step_start_time

            t = i / (config.Config.OBJECT_ROTATION_STEPS - 1)
            for body_id, start, displacement_World in motions:
                set_position(body_id, start + t * displacement_World)

            time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))

        for body_id, start, displacement_World in motions:
            set_position(body_id, start + displacement_World)
            if np.linalg.norm(displacement_World) > 1e-9:
                draw_arrow(start, displacement_World, colors[step % len(colors)],
                           length=np.linalg.norm(displacement_World), radius=0.015, tip_radius=0.035)

        wait_frames(pause_frames)

    for body_id, body_name, _ in bodies:
        label_ids[body_id] = show_label(f"{body_name}: done", body_id, label_ids[body_id])

    return [np.array(p.getBasePositionAndOrientation(body_id)[0]) for body_id, _, _ in bodies]


def set_pose(body_id, position, R):
    p.resetBasePositionAndOrientation(body_id, position, Rotation.from_matrix(R).as_quat())


def animate_rotation_about_point(body_id, pivot_World, R_apply, label):
    pos, quat = p.getBasePositionAndOrientation(body_id)
    pos_start, R_start = np.array(pos), np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    pivot_World = np.asarray(pivot_World, dtype=float)
    axis_World, angle_rad = rotation_to_axis_angle(R_apply)

    p.addUserDebugLine(pivot_World - 1.5 * axis_World, pivot_World + 1.5 * axis_World,
                       lineColorRGB=[1, 0, 1], lineWidth=3)
    draw_arrow(pivot_World, [0, 0, 1], [1, 0, 1], length=0.15, radius=0.02, tip_radius=0.05)  # pivot marker
    label_id = show_label(f"{label} ({np.degrees(angle_rad):.0f} deg)", body_id)

    for i in range(config.Config.OBJECT_ROTATION_STEPS):
        step_start_time = time.time()
        p.stepSimulation()
        elapsed_time = time.time() - step_start_time

        t = i / (config.Config.OBJECT_ROTATION_STEPS - 1)
        R_t = axis_angle_rotation(axis_World, t * angle_rad)
        set_pose(body_id, pivot_World + R_t @ (pos_start - pivot_World), R_t @ R_start)

        time.sleep(max(0, config.Config.SIM_TIME_STEP - elapsed_time))

    pos_final = pivot_World + R_apply @ (pos_start - pivot_World)
    R_final = R_apply @ R_start
    set_pose(body_id, pos_final, R_final)
    show_label(f"{label}: done", body_id, label_id)
    return pos_final, R_final
