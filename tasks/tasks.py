import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from config import config
from utils.rotations import (to_radians, intrinsic_rotation_steps, extrinsic_rotation_steps, rpy_to_rotations,
                             are_same_rotation, relative_rotation_angle_deg, shortest_rotation_World)
from utils.visualization import (get_rotation_matrix, set_orientation, show_label, wait_frames,
                                 animate_rotations, animate_axis_angle_rotation, animate_applied_rotations,
                                 draw_arrow, remove_arrows, animate_translations,
                                 set_pose, animate_rotation_about_point)


def task1(body1_id, body2_id):
    p.removeAllUserDebugItems()

    body1_rotations = to_radians([("Z", 130), ("X", 160), ("Y", -90)])
    body2_rotations = to_radians([("X", 100), ("Y", -100), ("Z", -100)])

    body1_steps_World = intrinsic_rotation_steps(body1_rotations, get_rotation_matrix(body1_id))
    body2_steps_World = intrinsic_rotation_steps(body2_rotations, get_rotation_matrix(body2_id))
    final_R1_World = body1_steps_World[-1]
    final_R2_World = body2_steps_World[-1]

    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 1 =====")
    print("final_R1_World =\n", final_R1_World)
    print("final_R2_World =\n", final_R2_World)
    print("R1^T @ R2 =\n", final_R1_World.T @ final_R2_World)
    same = are_same_rotation(final_R1_World, final_R2_World)
    if same:
        print("R1^T @ R2 == I -> the bodies have the SAME orientation")
    else:
        print(f"R1^T @ R2 != I -> the bodies have DIFFERENT orientations "
              f"(they differ by {relative_rotation_angle_deg(final_R1_World, final_R2_World):.2f} deg)")

    animate_rotations([
        (body1_id, "Body1", body1_rotations, body1_steps_World),
        (body2_id, "Body2", body2_rotations, body2_steps_World),
    ])
    p.addUserDebugText("SAME orientation" if same else "DIFFERENT orientations",
                       [0, 0, 2.6], textColorRGB=[0, 0.6, 0] if same else [0.8, 0, 0], textSize=2)

    return final_R1_World, final_R2_World


def task2(body1_id, body2_id):
    p.removeAllUserDebugItems()

    body1_rotations = rpy_to_rotations([100, 100, 10])
    body2_rotations = rpy_to_rotations([-105, 105, 0])

    body1_steps_World = extrinsic_rotation_steps(body1_rotations, np.eye(3))
    body2_steps_World = extrinsic_rotation_steps(body2_rotations, np.eye(3))
    R1_World = body1_steps_World[-1]
    R2_World = body2_steps_World[-1]

    axis_World, angle_rad = shortest_rotation_World(R1_World, R2_World)

    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 2 =====")
    print("R1_World =\n", R1_World)
    print("R2_World =\n", R2_World)
    print("R_rel = R2 @ R1^T =\n", R2_World @ R1_World.T)
    print(f"Shortest rotation: {np.degrees(angle_rad):.2f} deg around world axis {axis_World}")
    print(f"Axis in Body1 own frame: {R1_World.T @ axis_World}")

    set_orientation(body1_id, np.eye(3))
    set_orientation(body2_id, np.eye(3))
    animate_rotations([
        (body1_id, "Body1", body1_rotations, body1_steps_World),
        (body2_id, "Body2", body2_rotations, body2_steps_World),
    ], extrinsic=True)
    wait_frames(config.Config.FPS)

    label_id = animate_axis_angle_rotation(
        body1_id, R1_World, axis_World, angle_rad,
        f"Body1 shortest: {np.degrees(angle_rad):.1f} deg around magenta axis")
    show_label("Body1 now has Body2 orientation", body1_id, label_id)

    return axis_World, angle_rad


def task3(body1_id, body2_id):
    p.removeAllUserDebugItems()
    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 3 =====")

    bodies = []
    for body_id, body_name, rpy_deg in ((body1_id, "Body1", [5, 25, 20]), (body2_id, "Body2", [30, 90, 0])):
        rpy_rad = [angle_rad for _, angle_rad in rpy_to_rotations(rpy_deg)]
        rotation = Rotation.from_euler("xyz", rpy_rad)
        R_World = rotation.as_matrix()
        quat = rotation.as_quat()

        yxz_rad = rotation.as_euler("YXZ")
        own_rotations = list(zip("YXZ", yxz_rad))

        print(f"{body_name}: RPY = {rpy_deg} deg")
        print(f"RotMat{body_name[-1]} =\n", R_World)
        print(f"Quat{body_name[-1]} [x, y, z, w] = {quat}")
        print(f"Own Y, X, Z angles = {np.degrees(yxz_rad)} deg")

        bodies.append((body_id, body_name, own_rotations, intrinsic_rotation_steps(own_rotations, np.eye(3)), quat))

    for body_id, _, _, _, _ in bodies:
        set_orientation(body_id, np.eye(3))
    animate_rotations([body[:4] for body in bodies])

    for body_id, body_name, _, _, quat in bodies:
        show_label(f"Quat{body_name[-1]} = [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]",body_id, height=1.6)

    return [quat for *_, quat in bodies]


def task4(body1_id, body2_id):
    p.removeAllUserDebugItems()
    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 4 =====")

    rpy1_rad = [angle_rad for _, angle_rad in rpy_to_rotations([45, -45, 0])]
    rpy2_rad = [angle_rad for _, angle_rad in rpy_to_rotations([30, 0, 30])]
    RotMat1 = Rotation.from_euler("xyz", rpy1_rad).as_matrix()
    Quat2 = Rotation.from_euler("xyz", rpy2_rad).as_quat()  # [x, y, z, w]
    RotMat2 = Rotation.from_quat(Quat2).as_matrix()

    print("RotMat1 (RPY = [45, -45, 0]) =\n", RotMat1)
    print("Quat2 (RPY = [30, 0, 30]) [x, y, z, w] =", Quat2)
    print("RotMat2 (from Quat2) =\n", RotMat2)

    final_R1_World, final_R2_World = animate_applied_rotations([
        (body1_id, "Body1", [("RotMat2", RotMat2), ("RotMat1^T", RotMat1.T)], RotMat1),
        (body2_id, "Body2", [("RotMat2^T", RotMat2.T), ("RotMat2", RotMat2)], RotMat2),
    ])

    print("Body1 final = RotMat1^T @ RotMat2 @ RotMat1 =\n", final_R1_World)
    print("Body2 final = RotMat2 @ RotMat2^T @ RotMat2 =\n", final_R2_World)

    for body_id, body_name, R_initial, R_final in ((body1_id, "Body1", RotMat1, final_R1_World),
                                                   (body2_id, "Body2", RotMat2, final_R2_World)):
        unchanged = are_same_rotation(R_initial, R_final)
        if unchanged:
            text = f"{body_name}: orientation NOT changed"
        else:
            text = f"{body_name}: orientation CHANGED by {relative_rotation_angle_deg(R_initial, R_final):.2f} deg"
        print(text)
        show_label(text, body_id, color=(0, 0.6, 0) if unchanged else (0.8, 0, 0), height=1.6)

    return final_R1_World, final_R2_World


def task5(body1_id, body2_id):
    p.removeAllUserDebugItems()
    remove_arrows()
    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 5 =====")

    body1_rotations = to_radians([("Z", 165), ("X", -20), ("Y", -30)])
    body1_steps_World = intrinsic_rotation_steps(body1_rotations, np.eye(3))
    x_axes_World = [R_World[:, 0] for R_World in body1_steps_World]

    print("Own X axis at start:", x_axes_World[0])
    for step, ((axis, angle_rad), x_axis_World) in enumerate(zip(body1_rotations, x_axes_World[1:]), start=1):
        print(f"After step {step} ({np.degrees(angle_rad):.0f} deg around own {axis}): own X = {x_axis_World}")

    body1_pos, _ = p.getBasePositionAndOrientation(body1_id)
    colors = [[0.5, 0.5, 0.5], [1, 0.5, 0], [0.8, 0, 0.8], [0, 0.6, 0.6]]

    def show_x_axis(step):
        # Arrow is longer (2 m) than the model's red X axis (1 m), so its end and ball stick out of the body
        x_axis_World = x_axes_World[step]
        end = draw_arrow(body1_pos, x_axis_World, colors[step])
        p.addUserDebugText(f"X{step} = [{x_axis_World[0]:.3f}, {x_axis_World[1]:.3f}, {x_axis_World[2]:.3f}]",
                           end + [0, 0, 0.15 + 0.2 * step], textColorRGB=colors[step], textSize=1.5)

    set_orientation(body1_id, np.eye(3))
    set_orientation(body2_id, np.eye(3))
    show_label("Body2: world frame (reference)", body2_id)
    show_x_axis(0)
    animate_rotations([(body1_id, "Body1", body1_rotations, body1_steps_World)],
                      after_step=lambda step: show_x_axis(step + 1))

    return x_axes_World[1:]


def task6(body1_id, body2_id):
    p.removeAllUserDebugItems()
    remove_arrows()
    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 6 =====")

    X1, X2, X3, X4 = -0.1, -0.2, -0.3, -0.4

    set_orientation(body1_id, np.eye(3))
    set_orientation(body2_id, np.eye(3))
    R1_World = get_rotation_matrix(body1_id)
    R2_World = get_rotation_matrix(body2_id)
    world_x, world_y = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])

    body1_moves = [(f"{X1} m along world X", X1 * world_x),
                   (f"{X2} m along own X", X2 * R1_World[:, 0])]
    body2_moves = [(f"{X3} m along world Y", X3 * world_y),
                   (f"{X4} m along own Y", X4 * R2_World[:, 1])]

    body1_start = np.array(p.getBasePositionAndOrientation(body1_id)[0])
    body2_start = np.array(p.getBasePositionAndOrientation(body2_id)[0])

    body1_final, body2_final = animate_translations([
        (body1_id, "Body1", body1_moves),
        (body2_id, "Body2", body2_moves),
    ])
    body1_displacement = body1_final - body1_start
    body2_displacement = body2_final - body2_start

    print(f"Body1: start {body1_start} -> final {body1_final}, displacement {body1_displacement}")
    print(f"Body2: start {body2_start} -> final {body2_final}, displacement {body2_displacement}")

    same_position = np.allclose(body1_final, body2_final)
    same_displacement = np.allclose(body1_displacement, body2_displacement)
    print("Final positions match" if same_position else
          f"Final positions DO NOT match (distance {np.linalg.norm(body1_final - body2_final):.4f} m)")
    print("Displacements match" if same_displacement else "Displacements DO NOT match")

    p.addUserDebugText("Final positions MATCH" if same_position else "Final positions DO NOT match",
                       [0, 0, 2.6], textColorRGB=[0, 0.6, 0] if same_position else [0.8, 0, 0], textSize=2)

    return body1_final, body2_final


def task7(body1_id, body2_id):
    p.removeAllUserDebugItems()
    remove_arrows()
    np.set_printoptions(precision=4, suppress=True)
    print("===== Task 7 =====")

    pos1, quat1 = np.array([0.6, 0, 0]), np.array([0, 0, 0.707, 0.707])
    pos2, quat2 = np.array([0, 0.2, 0]), np.array([0, 0, 0.707, 0.707])
    R1_start = Rotation.from_quat(quat1).as_matrix()
    R2_start = Rotation.from_quat(quat2).as_matrix()

    rpy1_rad = [angle_rad for _, angle_rad in rpy_to_rotations([0, 0, 90])]
    rpy2_rad = [angle_rad for _, angle_rad in rpy_to_rotations([0, 0, -90])]
    RotMat1 = Rotation.from_euler("xyz", rpy1_rad).as_matrix()
    RotMat2 = Rotation.from_euler("xyz", rpy2_rad).as_matrix()

    print("RotMat1 =\n", RotMat1)
    print("RotMat2 =\n", RotMat2)

    set_pose(body1_id, pos1, R1_start)
    set_pose(body2_id, pos2, R2_start)
    for pos, text in ((pos1, "Body1 start"), (pos2, "Body2 start")):
        p.addUserDebugText(text, pos + [0, 0, -0.15], textColorRGB=[0.4, 0.4, 0.4], textSize=1.2)
    show_label("Body1: Transform1", body1_id)
    show_label("Body2: Transform2", body2_id, height=1.6)
    wait_frames(2 * config.Config.FPS)

    world_origin = np.zeros(3)
    pos1_final, R1_final = animate_rotation_about_point(body1_id, world_origin, RotMat1,
                                                        "Body1: RotMat1 around world origin")
    wait_frames(config.Config.FPS)

    #
    pos2_final, R2_final = animate_rotation_about_point(body2_id, pos1_final, RotMat2, "Body2: RotMat2 around Body1 center")

    print(f"Body1: {pos1} -> {pos1_final}")
    print("Body1 final orientation =\n", R1_final)
    print(f"Body1 final quat [x, y, z, w] = {Rotation.from_matrix(R1_final).as_quat()}")
    print(f"Body2: {pos2} -> {pos2_final}")
    print("Body2 final orientation =\n", R2_final)
    print(f"Body2 final quat [x, y, z, w] = {Rotation.from_matrix(R2_final).as_quat()}")

    for pos, text in ((pos1_final, "Body1 final"), (pos2_final, "Body2 final")):
        p.addUserDebugText(f"{text} {np.round(pos, 3).tolist()}", pos + [0, 0, -0.3],textColorRGB=[0, 0, 0.8], textSize=1.3)

    return (pos1_final, R1_final), (pos2_final, R2_final)
