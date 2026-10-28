import pybullet as p
import pybullet_data
import time
import os
import numpy as np

from config import config
from tasks import tasks

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)  # or p.DIRECT
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    path_to_model = os.path.join(config.Config.PROJECT_ROOT_DIR, "models/3d_axis.urdf")
    body1_id = p.loadURDF(path_to_model, basePosition=[1, 0, 1])
    body2_id = p.loadURDF(path_to_model, basePosition=[-1, 0, 1])
    p.changeDynamics(body1_id, -1, mass=0)
    p.changeDynamics(body2_id, -1, mass=0)
    p.setGravity(0, 0, -9.81)

    body1_pos, body1_quat = p.getBasePositionAndOrientation(body1_id)
    body2_pos, body2_quat = p.getBasePositionAndOrientation(body2_id)

    body1_rpy = p.getEulerFromQuaternion(body1_quat)
    body2_rpy = p.getEulerFromQuaternion(body2_quat)

    body1_rotation_matrix = p.getMatrixFromQuaternion(body1_quat)
    body1_rotation_matrix = np.array(body1_rotation_matrix).reshape(3, 3)

    body2_rotation_matrix = p.getMatrixFromQuaternion(body2_quat)
    body2_rotation_matrix = np.array(body2_rotation_matrix).reshape(3, 3)

    start_time = time.time()

    ###
    # tasks.task1(body1_id, body2_id)
    # tasks.task2(body1_id, body2_id)
    # tasks.task3(body1_id, body2_id)
    # tasks.task4(body1_id, body2_id)
    # tasks.task5(body1_id, body2_id)
    # tasks.task6(body1_id, body2_id)
    tasks.task7(body1_id, body2_id)
    ###

    # Keep the window open after the tasks are done
    while True:
        step_start_time = time.time()
        p.stepSimulation()
        elapsed_time = time.time() - step_start_time

        sleep_time = max(0, config.Config.SIM_TIME_STEP - elapsed_time)
        time.sleep(sleep_time)

    p.disconnect()
