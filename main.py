#!/usr/bin/env python3
from pybullet_tree_sim.pruning_environment import PruningEnv
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree

# from pybullet_tree_sim.utils.camera_helpers import get_fov_from_dfov

from pybullet_tree_sim.utils.trimesh_render import RenderScene
from pybullet_tree_sim.utils.rgb_stream_visualizer import RGBStreamVisualizer

import numpy as np
import secrets
import time
from zenlog import log

from scipy.spatial.transform import Rotation

import pprint as pp

from PIL import Image


def main():
    import trimesh
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    from pybullet_tree_sim.utils.mesh_objects import MeshObjects

    pbutils = PyBUtils(renders=True)
    robot_start_orientation = Rotation.from_euler("xyz", [0, 0, 180], degrees=True).as_quat()
    robot = Robot(
        pbclient=pbutils.pbclient,
        position=[0, 1, 0],
        orientation=robot_start_orientation,
        init_joint_angles=(
            0,  # linear slider
            -np.pi / 2 + np.pi / 4,
            -np.pi * 2 / 3,
            np.pi * 2 / 3,
            -np.pi,
            -np.pi / 2,
            0,
        ),
    )

    penv = PruningEnv(
        pbutils=pbutils,
        verbose=True,
    )

    tree = Tree(
        pbutils=pbutils,
        meshes_root="/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes",
        tree_id=1,
        tree_type="envy",
        namespace="LPy",
    )
    penv.activate_tree(tree=tree, include_support_posts=False)

    # # Run the sim a little just to get the environment properly loaded.
    for i in range(10):
        pbutils.pbclient.stepSimulation()
        time.sleep(0.1)

    # Create the PyRender scene for colors. Tree gets recolored in Tree() instantiation and exported to obj path
    pyr_scene = RenderScene(mesh=tree.recolored_mesh)

    camera_extrinsics = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof0"])
    camera_extrinsics = np.asarray(camera_extrinsics).reshape((4, 4), order="F")
    camera_pose = np.linalg.inv(camera_extrinsics)
    # camera_intrinsics = robot.sensors['tof0'].get_camera_intrinsics()
    # print(camera_extrinsics)
    # print(camera_intrinsics)
    pyr_scene.add_camera(camera=robot.sensors["tof0"], pose=camera_pose, mode="depth", camera_name="tof0")
    # pyr_scene.render_visual()

    rgb_viz = RGBStreamVisualizer(max_queue_size=5)
    rgb_viz.start()
    time.sleep(1)

    while True:
        try:
            # log.debug(f"{robot.sensors['tof0']}")
            tof0_view_matrix = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof0"])
            tof0_rgbd = robot.get_rgbd_at_cur_pose(
                camera=robot.sensors["tof0"],
                type="sensor",
                view_matrix=tof0_view_matrix,
            )
            camera_extrinsics = np.asarray(camera_extrinsics).reshape((4, 4), order="F")
            camera_pose = np.linalg.inv(camera_extrinsics)
            pyr_scene.update_camera_pose(camera_name="tof0", pose=camera_pose)
            color, depth = pyr_scene.render_optical_sensor(sensor=robot.sensors["tof0"])

            # print("COLOR:")
            # print(pp.pformat(color))
            # print("DEPTH:")
            # print(pp.pformat(depth))

            rgb_viz.update_frame(rgb_data=color)

            # tof1_view_matrix = robot.get_view_mat_at_curr_pose(camera=robot.sensors["tof1"])
            # tof1_rgbd = robot.get_rgbd_at_cur_pose(
            #     camera=robot.sensors["tof1"],
            #     type="sensor",
            #     view_matrix=tof1_view_matrix,
            # )
            # tof0_view_matrix = np.asarray(tof0_view_matrix).reshape((4, 4), order="F")
            # log.debug(f"{tof0_view_matrix[:3, 3]}")

            # Get user keyboard input, map to robot movement, camera capture, controller action
            keys_pressed = penv.get_key_pressed()
            move_action = robot.get_key_move_action(keys_pressed=keys_pressed)
            sensor_data = robot.get_key_sensor_action(keys_pressed=keys_pressed)

            joint_vels, jacobian = robot.calculate_joint_velocities_from_ee_velocity_dls(
                end_effector_velocity=move_action
            )
            robot.set_joint_velocities(joint_velocities=joint_vels)

            # Step simulation
            pbutils.pbclient.stepSimulation()
            time.sleep(0.001)
        except KeyboardInterrupt:
            rgb_viz.stop()
            break

    # # penv.deactivate_tree(tree_id_str="LPy_tree1")
    # penv.pbutils.pbclient.disconnect()
    return


if __name__ == "__main__":
    main()
