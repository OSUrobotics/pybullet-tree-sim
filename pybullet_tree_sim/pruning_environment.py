#!/usr/bin/env python3
from pybullet_tree_sim import CONFIG_PATH, MESHES_PATH, URDF_PATH, RGB_LABEL, ROBOT_URDF_PATH
from pybullet_tree_sim.tree import Tree, TreeException
from pybullet_tree_sim.utils.ur5_utils import UR5
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.xacro_utils as xutils
import pybullet_tree_sim.utils.yaml_utils as yutils

# from final_approach_controller.cut_point_rotate_axis_controller import CutPointRotateAxisController


from collections import defaultdict
import cv2
import glob
import gymnasium as gym
import numpy as np
import os
import skimage.draw as skdraw
import sys
import time
from typing import Optional, Tuple

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import modern_robotics as mr
from numpy.typing import ArrayLike

from zenlog import log


class PruningEnvException(Exception):
    def __init__(self, s, *args):
        super().__init__(args)
        self.s = s
        return

    def __str__(self):
        return f"{self.s}"


class PruningEnv(gym.Env):
    rgb_label = RGB_LABEL
    """
        PruningEnv is a custom environment that extends the gym.Env class from OpenAI Gym.
        This environment simulates a pruning task where a robot arm interacts with a tree.
        The robot arm is a UR5 arm and the tree is a 3D model of a tree.
        The environment is used to train a reinforcement learning agent to prune the tree.
    """

    _supports_and_post_xacro_path = os.path.join(URDF_PATH, "supports_and_post", "supports_and_post.urdf.xacro")
    _supports_and_post_urdf_path = os.path.join(URDF_PATH, "supports_and_post", "supports_and_post.urdf")
    _shapes_xacro_dir = os.path.join(URDF_PATH, "shapes")

    def __init__(
        self,
        # angle_threshold_perp: float = 0.52,
        # angle_threshold_point: float = 0.52,
        pbutils: PyBUtils,
        # cam_width: int = 424,
        # cam_height: int = 240,
        evaluate: bool = False,
        # distance_threshold: float = 0.05,
        max_steps: int = 1000,
        make_trees: bool = False,
        name: str = "PruningEnv",
        num_trees: int | None = None,
        renders: bool = False,
        tree_count: int = 10,
        # tree_urdf_path: str | None = None,
        # tree_obj_path: str | None = None,
        verbose: bool = True,
        load_robot: bool = True,
        robot_type: str = "ur5",
        robot_pos: ArrayLike = np.array([0, 0, 0]),
        robot_orientation: ArrayLike = np.array([0, 0, 0, 1]),
        # use_ik: bool = True,
        #
    ) -> None:
        """Initialize the Pruning Environment

        Parameters
        ----------
            cam_width (int): Pixel width of the camera
            cam_height (int): Pixel height of the camera
            evaluate (bool): Is this environment for evaluation
            max_steps (int): Maximum number of steps in a single run
            name (str): Name of the environment (default="PruningEnv")
            renders (bool): Whether to render the environment
            tree_count (int): Number of trees to be loaded
        """
        super().__init__()
        self.pbutils = pbutils

        # Pybullet GUI variables
        self.render_mode = "rgb_array"
        self.renders = renders
        self.eval = evaluate

        # Gym variables
        self.name = name
        self.step_counter = 0
        self.global_step_counter = 0
        # self.max_steps = max_steps
        self.tree_count = tree_count
        self.is_goal_state = False

        # sensor types
        self.sensor_attributes = {}  # TODO: Load sensor types from config files
        camera_configs_path = os.path.join(CONFIG_PATH, "camera")
        camera_configs_files = glob.glob(os.path.join(camera_configs_path, "*.yaml"))
        for file in camera_configs_files:
            yamlcontent = yutils.load_yaml(file)
            for key, value in yamlcontent.items():
                self.sensor_attributes[key] = value

        tof_configs_path = os.path.join(CONFIG_PATH, "tof")
        tof_configs_files = glob.glob(os.path.join(tof_configs_path, "*.yaml"))
        for file in tof_configs_files:
            yamlcontent = yutils.load_yaml(file)
            for key, value in yamlcontent.items():
                self.sensor_attributes[key] = value

        # log.warning(self.sensor_attributes)

        # self.cam_width = cam_width
        # self.cam_height = cam_height
        # self.cam_pan = 0
        # self.cam_tilt = 0
        # # self.cam_xyz_offset = np.zeros(3)
        # self.cam_xyz_offset = np.array([0, 0, 0])

        # # Camera stuff
        # self.pixel_coords = np.array(list(np.ndindex((cam_width, cam_height))), dtype=int)  # C-style
        # # Find the pixel coordinates in the film plane. Bin, offset, normalize, then scale to [-1, 1]
        # self.film_plane_coords = np.zeros((cam_width, cam_height, 2), dtype=float)
        # self.film_plane_coords = 2 * np.divide(
        #     np.subtract(np.add(self.pixel_coords, [0.5, 0.5]), [cam_width / 2, cam_height / 2]), [cam_width, cam_height]
        # )

        self.verbose = verbose

        self.collision_object_ids = {
            "SPUR": None,
            "TRUNK": None,
            "BRANCH": None,
            "WATER_BRANCH": None,
            "SUPPORT": None,
        }

        # Tree parameters
        # self.tree_goal_pos = None
        # self.tree_goal_or = None
        # self.tree_goal_normal = None
        # self.tree_urdf_path: str | None = None
        # self.tree_pos = np.zeros(3, dtype=float)
        # self.tree_orientation = np.zeros(3, dtype=float)
        # self.tree_scale: float = 1.0

        self.trees = {}
        self.debouce_time = 0.5
        self.last_button_push_time = time.time()

        # UR5 Robot
        if load_robot:
            self.ur5 = self.load_robot(
                type=robot_type, robot_pos=robot_pos, robot_orientation=robot_orientation, randomize_pose=False
            )
        return

    def load_robot(self, type: str, robot_pos: ArrayLike, robot_orientation: ArrayLike, randomize_pose: bool = False):
        """Load a robot into the environment. Currently only UR5 is supported. TODO: Add Panda"""
        type = type.strip().lower()
        if type == "ur5":
            log.info("Loading UR5 Robot")
            robot = UR5(
                con=self.pbutils.pbclient,
                robot_urdf_path=ROBOT_URDF_PATH,
                pos=robot_pos,
                orientation=robot_orientation,
                randomize_pose=randomize_pose,
                verbose=self.verbose,
            )
        else:
            raise NotImplementedError(f"Robot type {type} not implemented")
        return robot

    def load_tree(  # TODO: Clean up Tree init vs create_tree, probably not needed. Too many file checks.
        self,
        pbutils: PyBUtils,
        scale: float,
        position: ArrayLike = [0, 0, 0],
        orientation: ArrayLike = [0, 0, 0, 1],
        tree_id: int = None,
        tree_type: str = None,
        tree_namespace: str = "",
        tree_urdf_path: str | None = None,
        save_tree_urdf: bool = False,
        randomize_pose: bool = False,
    ) -> None:
        if randomize_pose:
            ...

        # If user supplied URDF file, check if it exists
        if tree_urdf_path is not None:
            if not os.path.exists(tree_urdf_path):
                raise OSError(
                    f"There do not seem to be any files of that name, please check your path. Given path was {tree_urdf_path}"
                )

        # Get tree object
        tree = Tree.create_tree(
            pbutils=pbutils,
            scale=scale,
            position=position,
            orientation=orientation,
            tree_id=tree_id,
            tree_type=tree_type,
            namespace=tree_namespace,
            tree_urdf_path=tree_urdf_path,
            randomize_pose=randomize_pose,
        )

        # tree_id_str = f"{tree_namespace}{tree_type}_tree{tree_id}"
        urdf_path = os.path.join(URDF_PATH, "trees", tree_type, "generated", f"{tree.id_str}.urdf")

        # Add tree to dict of trees
        self.trees[tree.id_str] = tree
        return

    def activate_tree(
        self, tree: Tree | None = None, tree_id_str: str | None = None, include_support_posts: bool = True
    ) -> None:
        """Activate a tree by object or by tree_id_str. Can include support posts. Must provide either a Tree or tree_id_str.
        @param tree (Tree/None): Tree object to be activated into the pruning environment.
        @param tree_id_str (str/None): String including the identification characteristics of the tree.
        @return None
        """

        if tree is None and tree_id_str is None:
            raise TreeException("Parameters 'tree' and 'tree_id_str' cannot both be None")

        if tree is None and tree_id_str is not None:
            try:
                tree = self.trees[tree_id_str]
            except KeyError as e:
                raise TreeException(f"{e}: Tree with ID {tree_id_str} not found")

        if tree is not None:
            if self.verbose:
                log.info("Activating tree")
            tree.pyb_tree_id = self.pbutils.pbclient.loadURDF(tree.urdf_path, useFixedBase=True)
            log.info(f"Tree {tree.id_str} activated with PyBID {tree.pyb_tree_id}")

        if include_support_posts:
            self.activate_support_posts(associated_tree=tree)
        return

    def deactivate_tree(self, tree: Tree | None = None, tree_id_str: str | None = None) -> None:
        """Deactivate a tree by object or by tree_id_str"""
        if tree is None and tree_id_str is None:
            raise TreeException("Parameters 'tree' and 'tree_id_str' cannot both be None")

        if tree is None and tree_id_str is not None:
            try:
                tree = self.trees[tree_id_str]
            except KeyError as e:
                raise TreeException(f"{e}: Tree with ID {tree_id_str} not found")

        if tree is not None:
            try:
                self.pbutils.pbclient.removeBody(tree.pyb_tree_id)
                log.info(f"Tree {tree.id_str} with PyBID {tree.pyb_tree_id} deactivated")
            except Exception as e:
                log.error(f"Error deactivating tree: {e}")
        return

    def activate_support_posts(
        self, associated_tree: Tree, position: ArrayLike | None = None, orientation: ArrayLike | None = None
    ) -> None:
        # Load support posts
        if position is None:
            position = [associated_tree.pos[0], associated_tree.pos[1] - 0.05, 0.0]
        if orientation is None:
            orientation = Rotation.from_euler("xyz", [np.pi / 2, 0, np.pi / 2]).as_quat()

        if not os.path.exists(
            self._supports_and_post_urdf_path
        ):  # TODO: change this -- it'll take a improperly positioned file and keep using it.'
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=self._supports_and_post_xacro_path, mappings=None
            ).toprettyxml()  # TODO: add mappings
            xutils.save_urdf(urdf_content=urdf_content, urdf_path=self._supports_and_post_urdf_path)

        support_post_id = self.pbutils.pbclient.loadURDF(
            fileName=self._supports_and_post_urdf_path, basePosition=position, baseOrientation=orientation
        )
        log.info(f"Supports and post activated with PyBID {support_post_id}")
        self.collision_object_ids["SUPPORT"] = support_post_id
        return

    def deactivate_support_posts(self) -> None:
        # try:
        #     self.pbutils.pbclient.removeBody(self.collision_object_ids["SUPPORT"])
        #     log.info(f"Supports and post deactivated")
        # except Exception as e:
        #     log.error(f"Error deactivating supports and post: {e}")
        return

    def reset_environment(self) -> None:
        """TODO"""
        # self.pbutils.pbclient.resetSimulation()
        return

    def activate_shape(
        self, shape: str, position: ArrayLike | None = None, orientation: ArrayLike | None = None, **kwargs
    ) -> None:
        """Activate a generic cylinder object in the environment.
        @param shape (str): shape type. Currently supported options are: [cylinder]
        @param position (ArrayLike): Vector containing the xyz position of the base. Cylinder default position is the center of the cylinder.
        @param orientation (ArrayLike): Vector containing the roll-pitch-yaw of the base about the world origin.
        @param radius (float, Optional): if
        @return None
        """
        log.warning(locals())
        shape = shape.strip().lower()
        shape_xacro_path = os.path.join(self._shapes_xacro_dir, shape, f"{shape}.urdf.xacro")
        shape_urdf_path = os.path.join(self._shapes_xacro_dir, shape, f"{shape}.urdf")

        shape_mappings = {}
        for key, value in kwargs.items():
            shape_mappings[key] = str(value)

        # shape position is the center of the shape, move up by half the height
        if position is None:
            position = [0, 0, float(shape_mappings["height"]) / 2]
        if orientation is None:
            orientation = [0, 0, 0, 1]
        else:
            orientation = Rotation.from_euler("xyz", orientation).as_quat()

        # print(position)

        # if shape == "cylinder":
        #     radius = kwargs.get('radius')
        #     height = kwargs.get('height')

        urdf_content = xutils.load_urdf_from_xacro(xacro_path=shape_xacro_path, mappings=shape_mappings).toprettyxml()
        xutils.save_urdf(urdf_content=urdf_content, urdf_path=shape_urdf_path)

        shape_id = self.pbutils.pbclient.loadURDF(
            fileName=shape_urdf_path, basePosition=position, baseOrientation=orientation
        )
        log.info(f"{shape.title()} loaded with PyBID {shape_id}")

        return

    def deproject_pixels_to_points(
        self, camera, data: np.ndarray, view_matrix: np.ndarray, return_frame: str = "world"
    ) -> np.ndarray:
        """Compute frame XYZ from image XY and measured depth. Default frame is 'world'.
        (pixel_coords -- [u,v]) -> (film_coords -- [x,y]) -> (camera_coords -- [X, Y, Z]) -> (world_coords -- [U, V, W])

        https://ksimek.github.io/2013/08/13/intrinsic/
        https://gachiemchiep.github.io/cheatsheet/camera_calibration/
        https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed

        @param data: nx1 array of depth values, Fortran order (column-first)
        @param view_matrix: 4x4 matrix (world -> camera transform)
        @param return_frame: str, either 'camera' or 'world'

        @return: nx4 array of world XYZ coordinates
        """

        # log.debug(f"View matrix:\n{view_matrix}")

        # Flip the y and z axes to convert from OpenGL camera frame to standard camera frame.
        # https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed
        # https://github.com/bitlw/LearnProjMatrix/blob/main/doc/OpenGL_Projection.md#introduction
        # view_matrix[1:3, :] = -view_matrix[1:3, :]
        proj_matrix = np.asarray(camera.depth_proj_mat).reshape([4, 4], order="F")

        # rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type='robot', view_matrix=view_matrix)
        # data = depth.reshape((self.cam_width * self.cam_height, 1), order="F")

        # Get camera intrinsics from projection matrix. If square camera, these should be the same.
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]  # if square camera, these should be the same

        # Get camera coordinates from film-plane coordinates. Scale, add z (depth), then homogenize the matrix.
        cam_coords = np.divide(np.multiply(camera.depth_film_coords, data), [fx, fy])
        cam_coords = np.concatenate((cam_coords, data, np.ones((camera.depth_width * camera.depth_height, 1))), axis=1)

        if return_frame.strip().lower() == "camera":
            return cam_coords

        world_coords = (mr.TransInv(view_matrix) @ cam_coords.T).T

        plot = False
        if plot:
            self.debug_plots(
                camera=camera, data=data, cam_coords=cam_coords, world_coords=world_coords, view_matrix=view_matrix
            )

        return world_coords

    def get_cam_to_frame_coords(
        self, cam_coords: np.ndarray, start_frame: str, end_frame: str = "world", view_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """Convert camera coordinates to other frame coordinates. Default is world.
        @param cam_coords: nx4 array of camera XYZ coordinates
        @param start_frame: str
        @param end_frame: str (default = 'world')
        @param view_matrix: 4x4 matrix (world -> camera transform)

        @return: nx4 array of world XYZ coordinates
        """
        end_frame = end_frame.strip().lower()
        if end_frame == "world" and view_matrix is None:
            raise ValueError("View matrix required for world frame conversion.")
        elif end_frame == "world":
            return (mr.TransInv(view_matrix) @ cam_coords.T).T

        start_frame = start_frame.strip().lower()
        end_frame_coords = (mr.TransInv(self.ur5.static_frames[f"{start_frame}_to_{end_frame}"]) @ cam_coords.T).T

        return end_frame_coords

    def compute_deprojected_point_mask(self):
        # TODO: Make this function nicer
        # Function. Be Nice.
        # """Find projection stuff in 'treefitting'. Simpole depth to point mask conversion."""
        # point_mask = np.zeros((self.pbutils.cam_height, self.pbutils.cam_width), dtype=np.float32)

        # proj_matrix = np.asarray(self.pbutils.proj_mat).reshape([4, 4], order="F")
        # view_matrix = np.asarray(
        # self.ur5.get_view_mat_at_curr_pose(pan=self.cam_pan, tilt=self.cam_tilt, xyz_offset=self.cam_xyz_offset)
        # ).reshape([4, 4], order="F")
        # projection = (
        #     proj_matrix
        #     @ view_matrix
        #     @ np.array([0.5, 0.5, 1, 1])
        #     # @ np.array([self.tree_goal_pos[0], self.tree_goal_pos[1], self.tree_goal_pos[2], 1])
        # )

        # # Normalize by w -> homogeneous coordinates
        # projection = projection / projection[3]
        # # log.info(f"View matrix: {view_matrix}")
        # log.info(f"Projection: {projection}")
        # # if projection within 1,-1, set point mask to 1
        # if projection[0] < 1 and projection[0] > -1 and projection[1] < 1 and projection[1] > -1:
        #     projection = (projection + 1) / 2
        #     row = self.pbutils.cam_height - 1 - int(projection[1] * (self.pbutils.cam_height))
        #     col = int(projection[0] * self.pbutils.cam_width)
        #     radius = 5  # TODO: Make this a variable proportional to distance
        #     # modern scikit uses a tuple for center
        #     rr, cc = skdraw.disk((row, col), radius)
        #     print(rr, cc)
        #     point_mask[
        #         np.clip(0, rr, self.pbutils.cam_height - 1), np.clip(0, cc, self.pbutils.cam_width - 1)
        #     ] = 1  # TODO: This is a hack, numbers shouldnt exceed max and min anyways

        # # resize point mask to algo_height, algo_width
        # point_mask_resize = cv2.resize(point_mask, dsize=(self.algo_width, self.algo_height))
        # point_mask = np.expand_dims(point_mask_resize, axis=0).astype(np.float32)
        return point_mask

    def is_reachable(self, vertex: Tuple[np.ndarray], base_xyz: np.ndarray) -> bool:
        # if vertex[3] != "SPUR":
        #     return False
        ur5_base_pos = np.array(base_xyz)

        # Meta condition
        dist = np.linalg.norm(ur5_base_pos - vertex[0], axis=-1)

        if dist >= 0.98:  # TODO: is this for the UR5? Should it be from a parameter file?
            return False

        j_angles = self.ur5.calculate_ik(vertex[0], None)
        # env.ur5.set_joint_angles(j_angles)
        # for _ in range(100):
        #     pyb.con.stepSimulation()
        # ee_pos, _ = env.ur5.get_current_pose(env.ur5.end_effector_index)
        # dist = np.linalg.norm(np.array(ee_pos) - vertex[0], axis=-1)
        # if dist <= 0.03:
        #     return True

        return False

    def get_reachable_points(self, tree: Tree, env, pyb):
        # self.reachable_points = list(filter(lambda x: self.is_reachable(x, env, pyb), self.vertex_and_projection))
        # np.random.shuffle(self.reachable_points)
        # print("Number of reachable points: ", len(self.reachable_points))
        # if len(self.reachable_points) < 1:
        #     print("No points in reachable points", self.urdf_path)
        #     # self.reset_tree()

        return self.reachable_points

    def get_key_pressed(self, relevant=None) -> list:
        """Return the keys pressed by the user."""
        keys_pressed = []
        events = self.pbutils.pbclient.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            keys_pressed.append(key)
        return keys_pressed

    def get_key_action(self, keys_pressed: list) -> np.ndarray:
        """Return an action based on the keys pressed."""
        action = np.array([0.0, 0.0, 0, 0.0, 0.0, 0.0])
        if keys_pressed:
            if ord("a") in keys_pressed:
                action[0] += 0.01
            if ord("d") in keys_pressed:
                action[0] += -0.01
            if ord("s") in keys_pressed:
                action[1] += 0.01
            if ord("w") in keys_pressed:
                action[1] += -0.01
            if ord("q") in keys_pressed:
                action[2] += 0.01
            if ord("e") in keys_pressed:
                action[2] += -0.01
            if ord("z") in keys_pressed:
                action[3] += 0.01
            if ord("c") in keys_pressed:
                action[3] += -0.01
            if ord("x") in keys_pressed:
                action[4] += 0.01
            if ord("v") in keys_pressed:
                action[4] += -0.01
            if ord("r") in keys_pressed:
                action[5] += 0.05
            if ord("f") in keys_pressed:
                action[5] += -0.05
            if ord("p") in keys_pressed:
                if time.time() - self.last_button_push_time > self.debouce_time:
                    # Get view and projection matrices
                    view_matrix = self.ur5.get_view_mat_at_curr_pose(pan=0, tilt=0, xyz_offset=[0, 0, 0])
                    log.warning(f"button p pressed")
                    rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type="robot", view_matrix=view_matrix)
                    # log.debug(f'depth:\n{depth}')
                    depth = -1 * depth.reshape((self.cam_width * self.cam_height, 1), order="F")
                    world_points = self.deproject_pixels_to_points(
                        data=depth, view_matrix=np.asarray(view_matrix).reshape([4, 4], order="F")
                    )
                    # log.debug(f"world_points: {world_points}")
                    self.last_button_push_time = time.time()
                else:
                    log.warning("debouce time not yet reached")
            if ord("o") in keys_pressed:
                pass
            if ord("t") in keys_pressed:
                # env.force_time_limit()
                infos = {}
                infos["TimeLimit.truncated"] = True
                self.reset_environment()  # TODO: Write
                # set_goal_callback._update_tree_properties()
                # env.is_goal_state = True

        else:
            action = np.array([0.0, 0.0, 0, 0.0, 0.0, 0.0])
            keys_pressed = []
        return action

    def run_sim(self) -> int:

        return 0


def main():
    # data = np.zeros((cam_width, cam_height), dtype=float)
    # generator = np.random.default_rng(seed=secrets.randbits(128))
    # data[0,0] = 0.31
    # data[:, 3:5] = tuple(generator.uniform(0.31, 0.35, (cam_height, 2)))

    # start = 0.31
    # stop = 0.35
    # data[:, 3:5] = np.array([np.arange(start, stop, (stop - start) / 8), np.arange(start, stop, (stop - start) / 8)]).T
    # data[-1, 3] = 0.31
    # data = data.reshape((cam_width * cam_height, 1), order="F")

    # log.warning(f"joint angles: {penv.ur5.get_joint_angles()}")
    return


if __name__ == "__main__":
    main()
