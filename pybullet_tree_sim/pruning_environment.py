#!/usr/bin/env python3
from xml.sax.xmlreader import XMLReader
from numpy.typing import NDArray
from plotly.graph_objs import Violin
from scipy.spatial.transform import Rotation
from std_msgs.msg import String
from pybullet_tree_sim import MESHES_PATH, URDF_PATH, RGB_LABEL, ROBOT_URDF_PATH
from pybullet_tree_sim.tree import Tree, TreeException
from pybullet_tree_sim.utils.ur5_utils import UR5
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.xacro_utils as xutils


from collections import defaultdict
import cv2
import glob
import gymnasium as gym
import numpy as np
import os
import skimage.draw as skdraw
import sys
from typing import Optional, Tuple
import xml

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

    def __init__(
        self,
        # angle_threshold_perp: float = 0.52,
        # angle_threshold_point: float = 0.52,
        pbutils: PyBUtils,
        cam_width: int = 424,
        cam_height: int = 240,
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

        # Camera params
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_pan = 0
        self.cam_tilt = 0
        # self.cam_xyz_offset = np.zeros(3)
        self.cam_xyz_offset = np.array([0, 0, 0])

        # Camera stuff
        self.pixel_coords = np.array(list(np.ndindex((cam_width, cam_height))), dtype=int)  # C-style
        # Find the pixel coordinates in the film plane. Bin, offset, normalize, then scale to [-1, 1]
        self.film_plane_coords = np.zeros((cam_width, cam_height, 2), dtype=float)
        self.film_plane_coords = 2 * np.divide(
            np.subtract(np.add(self.pixel_coords, [0.5, 0.5]), [cam_width / 2, cam_height / 2]), [cam_width, cam_height]
        )

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

        # UR5 Robot
        if load_robot:
            self.ur5 = self.load_robot(type=robot_type, robot_pos=robot_pos, robot_orientation=robot_orientation, randomize_pose=False)
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

    def load_tree( # TODO: Clean up Tree init vs create_tree, probably not needed. Too many file checks.
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

        if not os.path.exists(self._supports_and_post_urdf_path):
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=self._supports_and_post_xacro_path, mappings=None
            ).toprettyxml()  # TODO: add mappings
            xutils.save_urdf(urdf_content=urdf_content, urdf_path=self._supports_and_post_urdf_path)

        support_post = self.pbutils.pbclient.loadURDF(
            fileName=self._supports_and_post_urdf_path, basePosition=position, baseOrientation=orientation
        )
        log.info(f"Supports and post activated with PyBID {support_post}")
        self.collision_object_ids["SUPPORT"] = support_post
        return

    def deactivate_support_posts(self) -> None:
        # try:
        #     self.pbutils.pbclient.removeBody(self.collision_object_ids["SUPPORT"])
        #     log.info(f"Supports and post deactivated")
        # except Exception as e:
        #     log.error(f"Error deactivating supports and post: {e}")
        return

    def reset_environment(self) -> None:
        # self.pbutils.pbclient.resetSimulation()
        return


    def activate_cylinder(position: ArrayLike | None = None, orientation: ArrayLike | None = None) -> None:
        """Activate a generic cylinder object in the environment."""
        
        
        return
    
    
    def deproject_pixels_to_points(self, data: np.ndarray, view_matrix: np.ndarray) -> np.ndarray:
        """Compute world XYZ from image XY and measured depth.
        (pixel_coords -- [u,v]) -> (film_coords -- [x,y]) -> (camera_coords -- [X, Y, Z]) -> (world_coords -- [U, V, W])

        https://ksimek.github.io/2013/08/13/intrinsic/
        https://gachiemchiep.github.io/cheatsheet/camera_calibration/
        https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed

        @param data: 2D array of depth values

        TODO: change all `reshape` to `resize`
        """


        # log.debug(f"View matrix:\n{view_matrix}")

        # Flip the y and z axes to convert from OpenGL camera frame to standard camera frame.
        # https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed
        # https://github.com/bitlw/LearnProjMatrix/blob/main/doc/OpenGL_Projection.md#introduction
        view_matrix[1:3, :] = -view_matrix[1:3, :]
        proj_matrix = np.asarray(self.pbutils.proj_mat).reshape([4, 4], order="F")


        # rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type='robot', view_matrix=view_matrix)
        # data = depth.reshape((self.cam_width * self.cam_height, 1), order="F")


        # Get camera intrinsics from projection matrix
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]  # if square camera, these should be the same

        # Get camera coordinates from film-plane coordinates. Scale, add z (depth), then homogenize the matrix.
        cam_coords = np.divide(np.multiply(self.film_plane_coords, data), [fx, fy])
        cam_coords = np.concatenate((cam_coords, data, np.ones((self.cam_width * self.cam_height, 1))), axis=1)

        world_coords = (mr.TransInv(view_matrix) @ cam_coords.T).T

        plot = True
        if plot:
            self.debug_plots(data=data, cam_coords=cam_coords, world_coords=world_coords, view_matrix=view_matrix)

        return world_coords

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
        pressed_keys = []
        events = self.pbutils.pbclient.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

    def get_key_action(self, keys_pressed: list) -> np.ndarray:
        """Return an action based on the keys pressed."""
        action = np.array([0.,0.,0, 0., 0., 0.])
        # keys_pressed = self.get_key_pressed()
        if keys_pressed:
            # TODO: Make these values all sum so that multi-dof actions can be performed
            if ord('a') in keys_pressed:
                # action = np.array([0.01, 0, 0, 0, 0, 0])
                action[0] += 0.01
            elif ord('d') in keys_pressed:
                # action = np.array([-0.01, 0, 0, 0, 0, 0])
                action[0] += -0.01
            elif ord('s') in keys_pressed:
                # action = np.array([0, 0.01, 0, 0, 0, 0])
                action[1] += 0.01
            elif ord('w') in keys_pressed:
                # action = np.array([0, -0.01, 0, 0, 0, 0])
                action[1] += -0.01
            elif ord('q') in keys_pressed:
                # action = np.array([0, 0, 0.01, 0, 0, 0])
                action[2] += 0.01
            elif ord('e') in keys_pressed:
                # action = np.array([0, 0, -0.01, 0, 0, 0])
                action[2] += -0.01
            elif ord('z') in keys_pressed:
                # action = np.array([0, 0, 0, 0.01, 0, 0])
                action[3] += 0.01
            elif ord('c') in keys_pressed:
                # action = np.array([0, 0, 0, -0.01, 0, 0])
                action[3] += -0.01
            elif ord('x') in keys_pressed:
                # action = np.array([0, 0, 0, 0, 0.01, 0])
                action[4] += 0.01
            elif ord('v') in keys_pressed:
                # action = np.array([0, 0, 0, 0, -0.01, 0])
                action[4] += -0.01
            elif ord('r') in keys_pressed:
                # action = np.array([0, 0, 0, 0, 0, 0.05])
                action[5] += 0.05
            elif ord('f') in keys_pressed:
                # action = np.array([0, 0, 0, 0, 0, -0.05])
                action[5] += -0.05
            elif ord('p') in keys_pressed:
                # Get view and projection matrices
                view_matrix = np.asarray(self.ur5.get_view_mat_at_curr_pose(pan=0, tilt=0, xyz_offset=[0,0,0])).reshape([4, 4], order="F")
                rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type='robot', view_matrix=view_matrix)
                log.debug(depth)
                depth = depth.reshape((self.cam_width * self.cam_height, 1), order="F")
                world_points = self.deproject_pixels_to_points(data=depth, view_matrix=view_matrix)
                # log.debug(f"world_points: {world_points}")
            elif ord('t') in keys_pressed:
                # env.force_time_limit()
                infos = {}
                infos['TimeLimit.truncated'] = True
                self.reset_environment() # TODO: Write
                # set_goal_callback._update_tree_properties()
                # env.is_goal_state = True
            else:
                action = np.array([0.,0.,0, 0., 0., 0.])
                keys_pressed = []
        else:
            action = np.array([0.,0.,0, 0., 0., 0.])
            keys_pressed = []
        return action

    def debug_plots(self, data, cam_coords, world_coords, view_matrix):
        import plotly.graph_objects as go

        hovertemplate = "id: %{id}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>"

        _data = data.reshape([self.cam_width, self.cam_height], order="F")
        _data = _data.reshape((self.cam_width * self.cam_height, 1), order="C")
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=list(range(8)) * 8,
                    y=np.array([list(range(8))] * 8).T.flatten(order="C"),
                    z=_data.flatten(order="C"),
                    mode="markers",
                    ids=np.array([f"{i}" for i in range(self.cam_width * self.cam_height)])
                    .reshape((8, 8), order="C")
                    .flatten(order="F"),
                    hovertemplate=hovertemplate,
                )
            ]
        )
        fig.update_layout(
            title="Pixel Coordinates",
            scene=dict(
                aspectmode="cube",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.25, y=-1.25, z=1.25),
                ),
            ),
        )
        fig.show()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=self.film_plane_coords[:, 0],
                    y=self.film_plane_coords[:, 1],
                    z=data.flatten(order="F"),
                    mode="markers",
                    ids=[f"{i}" for i in range(self.cam_width * self.cam_height)],
                    hovertemplate=hovertemplate,
                )
            ]
        )
        fig.update_layout(
            title="Film Coordinates",
            scene=dict(
                aspectmode="cube",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.25, y=-1.25, z=1.25),
                ),
            ),
        )
        fig.show()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=cam_coords[:, 0],
                    y=cam_coords[:, 1],
                    z=cam_coords[:, 2],
                    mode="markers",
                    ids=[f"{i}" for i in range(self.cam_width * self.cam_height)],
                    hovertemplate=hovertemplate,
                )
            ]
        )  # reverse sign of z to match world coords
        fig.update_layout(
            title="Camera Coordinates",
            scene=dict(
                aspectmode="cube",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.25, y=-1.25, z=1.25),
                ),
            ),
        )
        fig.show()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=world_coords[:, 0],
                    y=world_coords[:, 1],
                    z=world_coords[:, 2],
                    name="tof_data",
                    mode="markers",
                    marker=dict(size=2),
                    ids=np.array([f"{i}" for i in range(self.cam_width * self.cam_height)]),
                    hovertemplate=hovertemplate,
                )
            ]
        )
        inv_view_matrix = mr.TransInv(view_matrix)
        fig.add_trace(
            go.Scatter3d(
                x=[inv_view_matrix[0, 3]],
                y=[inv_view_matrix[1, 3]],
                z=[inv_view_matrix[2, 3]],
                mode="markers",
                name="camera_origin",
                marker=dict(size=5),
            )
        )
        fig.update_layout(
            title="World Coordinates",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(range=[-1.0, 1.0]),
                yaxis=dict(range=[-0.1, 1.0]),
                zaxis=dict(range=[-0.0, 2.1]),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.5, y=-1.5, z=1.5),
                ),
            ),
        )
        fig.show()

        # log.warn(f"view_matrix: {view_matrix}")
        # log.warn(f"inv_view_matrix: {inv_view_matrix}")
        return

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
