#!/usr/bin/env python3

from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.camera_helpers import get_fov_from_dfov

import numpy as np
import secrets
import time
from zenlog import log


def main():
    # TODO: CLI args?
    cam_dfov = 65
    cam_height = 8
    cam_width = 8

    pbutils = PyBUtils(renders=True, cam_width=cam_width, cam_height=cam_height, dfov=cam_dfov)
    penv = PruningEnv(
        pbutils=pbutils, load_robot=True, robot_pos=[0, 1, 0], verbose=True, cam_width=cam_width, cam_height=cam_height
    )

    penv.activate_shape(shape="cylinder", radius=2 * 0.0254, height=2.0, orientation=[0, np.pi / 2, 0])
    # penv.load_tree(
    #     pbutils=pbutils,
    #     scale=1.0,
    #     tree_id=1,
    #     tree_type="envy",
    #     tree_namespace="LPy_",
    #     # tree_urdf_path=os.path.join(URDF_PATH, "trees", "envy", "generated", "LPy_envy_tree0.urdf"),
    #     save_tree_urdf=False,
    #     # randomize_pose=True
    # )
    # penv.activate_tree(tree_id_str="LPy_envy_tree1")

    # Run the sim a little just to get the environment properly loaded.
    for i in range(100):
        pbutils.pbclient.stepSimulation()
        time.sleep(0.1)

    # Simulation loop
    while True:
        try:
            view_matrix = penv.ur5.get_view_mat_at_curr_pose(0, 0, [0, 0, 0])
            rgbd = penv.pbutils.get_rgbd_at_cur_pose(type="robot", view_matrix=view_matrix)
            keys_pressed = penv.get_key_pressed()
            action = penv.get_key_action(keys_pressed=keys_pressed)
            action = action.reshape((6, 1))
            jv, jacobian = penv.ur5.calculate_joint_velocities_from_ee_velocity_dls(end_effector_velocity=action)
            penv.ur5.action = jv
            singularity = penv.ur5.set_joint_velocities(penv.ur5.action)
            penv.pbutils.pbclient.stepSimulation()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break

    # penv.deactivate_tree(tree_id_str="LPy_envy_tree1")
    penv.pbutils.pbclient.disconnect()
    return


if __name__ == "__main__":
    main()
