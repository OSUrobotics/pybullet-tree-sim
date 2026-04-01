#!/usr/bin/env python3
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.sensors.sensor import Sensor
from pybullet_tree_sim.sensors.sensor_map import SENSOR_TYPE_MAP
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.camera_helpers as ch
import pybullet_tree_sim.utils.xacro_utils as xutils
import pybullet_tree_sim.utils.yaml_utils as yutils
from pybullet_tree_sim import CONFIG_PATH, MESHES_PATH, URDF_PATH

import pybullet_tree_sim.plot as plot


import glob
from typing import Optional, Tuple
import modern_robotics as mr
import numpy as np
from pathlib import Path
import pybullet
import os
import time
import tempfile

import pprint as pp
from scipy.spatial.transform import Rotation

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Robot:

    _robot_configs_path = os.path.join(CONFIG_PATH, "robot")
    _robot_xacro_path = os.path.join(URDF_PATH, "robot", "robot.urdf.xacro")
    # _urdf_tmp_path = os.path.join(URDF_PATH, "tmp")

    def __init__(
        self,
        pbclient,
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        init_joint_angles: Optional[list] = None,
        randomize_pose=False,
        verbose=True,
    ) -> None:
        self.pbclient = pbclient
        self.verbose = verbose
        self.position = position
        self.orientation = orientation
        self.randomize_pose = randomize_pose  # TODO: This isn't set up anymore... fix
        self.init_joint_angles = (  # TODO: dynamically assign from defaults (i.e. if linear-slider is loaded in)
            (
                -np.pi / 2 + np.pi / 4,
                -np.pi * 2 / 3,
                np.pi * 2 / 3,
                -np.pi,
                -np.pi / 2,
                0,
            )
            if init_joint_angles is None
            else init_joint_angles
        )

        # Temporary directory for URDF
        self._urdf_tmp_dir = tempfile.mkdtemp(prefix="pybullet_tree_sim_robot_")

        # Robot setup
        self.robot = None

        # Sensors
        self.sensors = {}
        self.debounce_time = time.time()

        # Load robot URDF config
        self.robot_conf = {}
        self._generate_robot_urdf()
        self._setup_robot()
        self.num_joints = self.pbclient.getNumJoints(self.robot)
        self.robot_stack: list = self.robot_conf["robot_stack"]

        # Links
        self.links = self._get_links()
        self.robot_collision_filter_idxs = self._assign_collision_links()
        self.set_collision_filter(self.robot_collision_filter_idxs)
        self.tool0_link_idx = self._get_tool0_link_idx()
        self._assign_sensor_link_ids()

        # Joints
        self.joints = self._get_joints()
        # logger.info(f"Robot joints: {pp.pformat(self.joints)}")
        self.control_joints, self.control_joint_idxs = self._assign_control_joints(self.joints)
        self.reset_robot()

        # Robot action parameters
        self.action = None
        self.set_joint_angles_no_collision(self.init_joint_angles)
        self.pbclient.stepSimulation()
        return

    def _generate_robot_urdf(self) -> None:
        # Get robot params
        self.robot_conf.update(yutils.load_yaml(os.path.join(self._robot_configs_path, "robot.yaml")))
        self.robot_conf.update({"robot_stack_qty": str(len(self.robot_conf["robot_stack"]))})
        self.robot_conf.update(
            {
                "mesh_base_path": MESHES_PATH,
                "urdf_base_path": URDF_PATH,
            }
        )
        # Add the required urdf args from each element of the robot_stack config
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            robot_part = robot_part.strip().lower()
            # Assign parent frames
            if i == 0:
                self.robot_conf.update({f"parent{i}": "world"})
            else:
                self.robot_conf.update({f"parent{i}": self.robot_conf["robot_stack"][i - 1]})
            # Assign part frame ids
            self.robot_conf.update({f"robot_part{i}": self.robot_conf["robot_stack"][i]})
            # Add each robot part's config to the robot_conf
            part_conf = yutils.load_yaml(os.path.join(self._robot_configs_path, f"{robot_part}.yaml"))

            if part_conf is not None:
                if "sensors" in part_conf.keys():
                    part_tf_prefix_key = next(k for k in part_conf if k.endswith("tf_prefix"))
                    part_tf_prefix = part_conf[part_tf_prefix_key]

                    self.robot_conf.update({f"{robot_part}_sensor_qty": str(len(part_conf["sensors"].values()))})
                    for i, (sensor_name, sensor_metadata) in enumerate(part_conf["sensors"].items()):
                        _sensor = self._get_sensor(
                            sensor_type=sensor_metadata["type"],
                            sensor_model=sensor_metadata["model"],
                            sensor_name=sensor_name,
                            tf_frame=part_tf_prefix + "__" + sensor_metadata["tf_frame"],
                        )
                        self.sensors[sensor_name] = _sensor

                        # log.warn(pp.pformat(vars(_sensor)))

                        self.robot_conf.update(
                            {
                                f"sensor{i+1}_name": sensor_name,
                                f"sensor{i+1}_model": sensor_metadata["model"],
                                f"sensor{i+1}_position": sensor_metadata["position"],
                                f"sensor{i+1}_orientation": sensor_metadata["orientation"],
                                f"sensor{i+1}_mass": str(_sensor.mass),
                                f"sensor{i+1}_shape": _sensor.shape,
                                f"sensor{i+1}_dim_x": str(_sensor.dimensions[0]),
                                f"sensor{i+1}_dim_y": str(_sensor.dimensions[1]),
                                f"sensor{i+1}_dim_z": str(_sensor.dimensions[2]),
                                f"sensor{i+1}_base__sensing_unit_xyz_offset": " ".join(
                                    str(v) for v in _sensor.base__sensing_unit_xyz_offset
                                ),
                                f"sensor{i+1}_base__sensing_unit_rpy_offset": " ".join(
                                    str(v) for v in _sensor.base__sensing_unit_rpy_offset
                                ),
                            }
                        )

                self.robot_conf.update(part_conf)
            else:
                raise ValueError(f"Robot part {robot_part} not found in {self._robot_configs_path}")

        # log.warn(pp.pformat(self.robot_conf))
        # Generate URDF from mappings
        robot_urdf = xutils.load_urdf_from_xacro(
            xacro_path=self._robot_xacro_path,
            mappings=self.robot_conf,  # for some reason, this adds in the rest of the args from the xacro.
        )
        robot_urdf = robot_urdf.toprettyxml()

        # UR_description uses filename="package://<>" for meshes, and this doesn't work with pybullet
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            if robot_part.startswith("ur"):
                ur_absolute_mesh_path = "/opt/ros/humble/share/ur_description/meshes"
                robot_urdf = robot_urdf.replace(
                    f'filename="package://ur_description/meshes',
                    f'filename="{ur_absolute_mesh_path}',
                )
        # Save the generated URDF
        self.robot_urdf_path = os.path.join(self._urdf_tmp_dir, "robot.urdf")
        xutils.save_urdf(robot_urdf, urdf_path=self.robot_urdf_path)
        return

    def _setup_robot(self):
        if self.robot is not None:
            self.pbclient.removeBody(self.robot)
            del self.robot
        flags = self.pbclient.URDF_USE_SELF_COLLISION

        if self.randomize_pose:
            delta_pos = np.random.rand(3) * 0.0
            delta_orientation = pybullet.getQuaternionFromEuler(np.random.rand(3) * np.pi / 180 * 5)
        else:
            delta_pos = np.array([0.0, 0.0, 0.0])
            delta_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])

        self.position, self.orientation = self.pbclient.multiplyTransforms(
            self.position, self.orientation, delta_pos, delta_orientation
        )
        self.robot = self.pbclient.loadURDF(  # TODO: change to PyB_ID, this isn't a robot
            self.robot_urdf_path,
            self.position,
            self.orientation,
            flags=flags,
            useFixedBase=True,
        )
        self.num_joints = self.pbclient.getNumJoints(self.robot)

        # # self.init_pos_ee = self.get_current_pose(self.end_effector_index)
        # # self.init_pos_base = self.get_current_pose(self.base_index)
        # # self.init_pos_eebase = self.get_current_pose(self.success_link_index)
        # # self.action = np.zeros(len(self.init_joint_angles), dtype=np.float32)
        # self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        # # self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        # # base_pos, base_or = self.get_current_pose(self.base_index)
        return

    def _get_joints(self) -> dict:
        """Return a dict of joint information for the robot"""
        joints = {}
        self.base_index = 0
        for i in range(self.num_joints):
            info = self.pbclient.getJointInfo(self.robot, i)
            joint_name = info[1].decode("utf-8")
            joints.update(
                {
                    joint_name: {
                        "id": i,
                        "type": info[2],
                        "lower_limit": info[8],
                        "upper_limit": info[9],
                        "max_force": info[10],
                        "max_velocity": info[11],
                    }
                }
            )

        self.end_effector_index = 21  # TODO: Get dynamically

        return joints

    def _assign_control_joints(self, joints: dict) -> tuple[list]:
        """Get list of controllable joints from the joint dict by joint type"""
        control_joints = []
        control_joint_idxs = []
        self.control_joint_lower_limits = []
        self.control_joint_upper_limits = []
        self.control_joint_ranges = []
        for joint, joint_info in joints.items():
            if joint_info["type"] in (0, 1):  # (revolute, prismatic)
                control_joints.append(joint)
                control_joint_idxs.append(joint_info["id"])
                self.control_joint_lower_limits.append(joint_info["lower_limit"])
                self.control_joint_upper_limits.append(joint_info["upper_limit"])
                self.control_joint_ranges.append(joint_info["upper_limit"] - joint_info["lower_limit"])

        return control_joints, control_joint_idxs

    def _get_links(self) -> dict:
        links = {}
        for i in range(self.num_joints):
            info = self.pbclient.getJointInfo(self.robot, i)
            # log.debug(info)
            child_link_name = info[12].decode("utf-8")
            links.update({child_link_name: {"id": i, "tf_from_parent": info[14]}})
        return links

    def _assign_collision_links(self) -> list[int]:
        """Find tool0/base pairs, add to collision filter list.
        Requires that the robot part is ordered from base to tool0.

        TODO: Clean this up, there must be a better way.
        """
        robot_collision_filter_idxs = []
        for i, robot_part in enumerate(self.robot_conf["robot_stack"]):
            joint_info = self.pbclient.getJointInfo(self.robot, i)
            if i == 0:
                continue
            else:
                if (
                    robot_part + "__base" in self.links.keys()
                    and self.robot_conf["robot_stack"][i - 1] + "__tool0" in self.links.keys()
                ):
                    robot_collision_filter_idxs.append(
                        (
                            self.links[robot_part + "__base"]["id"],
                            self.links[self.robot_conf["robot_stack"][i - 1] + "__tool0"]["id"],
                        )
                    )
        return robot_collision_filter_idxs

    def _get_tool0_link_idx(self) -> int:
        """TODO: Clean up, find a better way?"""
        tool0_link_idx = None
        i = -1
        while tool0_link_idx is None:
            try:
                tool0_link_idx = self.links[self.robot_conf["robot_stack"][i] + "__tool0"]["id"]
            except KeyError:
                tool0_link_idx = None
                i -= 1
        return tool0_link_idx

    def _get_sensor(self, sensor_type: str, sensor_model: str, *args, **kwargs) -> Sensor:
        return SENSOR_TYPE_MAP[sensor_type](
            sensor_type=sensor_type.lower().strip(),
            sensor_model=sensor_model.lower().strip(),
            pbclient=self.pbclient,
            *args,
            **kwargs,
        )

    def _assign_sensor_link_ids(self) -> None:
        for sensor_name, sensor in self.sensors.items():
            sensor.tf_id = self.links[sensor.tf_frame + "_sensing_unit"]["id"]
        return

    def reset_robot(self, joint_angles: tuple = None) -> None:
        if self.robot is None:
            return
        self.init_joint_angles = (
            (
                0,  # linear slider
                -np.pi / 2 + np.pi / 4,
                -np.pi * 2 / 3,
                np.pi * 2 / 3,
                -np.pi,
                -np.pi / 2,
                0,
            )
            if joint_angles is None
            else joint_angles
        )
        self.set_joint_angles_no_collision(self.init_joint_angles)
        # self.pbclient.resetJointState() # TODO Fill out
        return

    def remove_robot(self):
        self.pbclient.removeBody(self.robot)
        self.robot = None
        return

    def set_joint_angles_no_collision(self, joint_angles) -> None:
        assert len(joint_angles) == len(self.control_joints)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            self.pbclient.resetJointState(self.robot, joint["id"], joint_angles[i], targetVelocity=0)
        return

    def set_joint_angles(self, joint_angles) -> None:
        """Set joint angles using pybullet motor control"""
        assert len(joint_angles) == len(self.control_joints)
        poses = []
        indices = []
        forces = []

        for i, name in enumerate(self.control_joints):
            # joint = self.robot_conf["joint_info"][name]
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indices.append(joint["id"])
            forces.append(joint["max_force"])

        self.pbclient.setJointMotorControlArray(
            self.robot,
            indices,
            self.pbclient.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces,
        )
        return

    def set_joint_velocities(self, joint_velocities) -> None:
        """Set joint velocities using pybullet motor control"""
        assert len(joint_velocities) == len(self.control_joints)
        velocities = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint["id"])
            forces.append(joint["max_force"])

        self.pbclient.setJointMotorControlArray(
            self.robot,
            indexes,
            controlMode=self.pbclient.VELOCITY_CONTROL,
            targetVelocities=joint_velocities,
        )
        return

    # TODO: Use proprty decorator for getters?
    def get_joint_velocities(self):
        j = self.pbclient.getJointStates(self.robot, self.control_joint_idxs)
        joints = tuple((i[1] for i in j))
        return joints  # type: ignore

    def get_joint_angles(self):
        """Return joint angles"""
        # print(self.robot_conf["control_joints"], self.controllable_joints_idxs)
        j = self.pbclient.getJointStates(self.robot, self.control_joint_idxs)
        joints = tuple((i[0] for i in j))
        return joints

    def get_current_pose(self, index):
        """Returns current pose of the index"""
        link_state = self.pbclient.getLinkState(self.robot, index, computeForwardKinematics=True)
        position, orientation = link_state[4], link_state[5]
        return position, orientation

    def get_current_vel(self, index):
        """Returns current pose of the index."""
        link_state = self.pbclient.getLinkState(
            self.robot,
            index,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
        )
        trans, ang = link_state[6], link_state[7]
        return trans, ang

    def get_condition_number(self):
        """Get condition number of the jacobian"""
        jacobian = self.calculate_jacobian()
        condition_number = np.linalg.cond(jacobian)
        return condition_number

    def calculate_ik(self, position, orientation):
        """Calculates joint angles from end effector position and orientation using inverse kinematics"""
        joint_angles = self.pbclient.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.end_effector_index,
            targetPosition=position,
            targetOrientation=orientation,
            jointDamping=[0.01] * len(self.control_joints),
            upperLimits=self.control_joint_upper_limits,
            lowerLimits=self.control_joint_lower_limits,
            jointRanges=self.control_joint_ranges,  # , restPoses=self.init_joint_angles
        )
        # print(self.control_joint_ranges)
        return joint_angles

    def calculate_jacobian(self):
        jacobian = self.pbclient.calculateJacobian(
            bodyUniqueId=self.robot,
            linkIndex=self.tool0_link_idx,
            localPosition=[0, 0, 0],
            objPositions=self.get_joint_angles(),
            objVelocities=[0] * len(self.control_joints),
            objAccelerations=[0] * len(self.control_joints),
        )
        jacobian = np.vstack(jacobian)
        return jacobian

    def calculate_joint_velocities_from_ee_velocity(self, end_effector_velocity):
        """Calculate joint velocities from end effector velocity using jacobian using least squares"""
        jacobian = self.calculate_jacobian()
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
        return joint_velocities, jacobian

    def calculate_joint_velocities_from_ee_velocity_dls(self, end_effector_velocity, damping_factor: float = 0.05):
        """Calculate joint velocities from end effector velocity using damped least squares"""
        jacobian = self.calculate_jacobian()
        identity_matrix = np.eye(jacobian.shape[0])
        damped_matrix = jacobian @ jacobian.T + (damping_factor**2) * identity_matrix
        damped_matrix_inv = np.linalg.inv(damped_matrix)
        dls_inv_jacobian = jacobian.T @ damped_matrix_inv
        joint_velocities = dls_inv_jacobian @ end_effector_velocity
        return joint_velocities, jacobian

    def set_collision_filter(self, robot_collision_filter_idxs) -> None:
        """Disable collision between pruner and arm"""
        for i in robot_collision_filter_idxs:
            self.pbclient.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 0)
        return

    def unset_collision_filter(self):
        """Enable collision between pruner and arm"""
        for i in self.robot_collision_filter_idxs:
            self.pbclient.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 1)
        return

    def disable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.pbclient.setCollisionFilterPair(self.robot, self.robot, i, j, 0)
        return

    def enable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.pbclient.setCollisionFilterPair(self.robot, self.robot, i, j, 1)
        return

    def check_collisions(self, collision_objects) -> Tuple[bool, dict]:
        """Check if there are any collisions between the robot and the environment
        Returns: Dictionary with information about collisions (Acceptable and Unacceptable)
        """
        collision_info = {
            "collisions_acceptable": False,
            "collisions_unacceptable": False,
        }

        collision_acceptable_list = ["SPUR", "WATER_BRANCH"]
        collision_unacceptable_list = ["TRUNK", "BRANCH", "SUPPORT"]
        for type in collision_acceptable_list:
            collisions_acceptable = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=collision_objects[type])
            if collisions_acceptable:
                for i in range(len(collisions_acceptable)):
                    if collisions_acceptable[i][-6] < 0:
                        collision_info["collisions_acceptable"] = True
                        break
            if collision_info["collisions_acceptable"]:
                break

        for type in collision_unacceptable_list:
            collisions_unacceptable = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=collision_objects[type])
            for i in range(len(collisions_unacceptable)):
                if collisions_unacceptable[i][-6] < 0:
                    collision_info["collisions_unacceptable"] = True
                    # break
            if collision_info["collisions_unacceptable"]:
                break

        if not collision_info["collisions_unacceptable"]:
            collisons_self = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=self.robot)
            collisions_unacceptable = collisons_self
            for i in range(len(collisions_unacceptable)):
                if collisions_unacceptable[i][-6] < -0.00:
                    collision_info["collisions_unacceptable"] = True
                    break
        if self.verbose > 1:
            print(f"DEBUG: {collision_info}")

        if collision_info["collisions_acceptable"] or collision_info["collisions_unacceptable"]:
            return True, collision_info

        return False, collision_info

    def check_success_collision(self, body_b) -> bool:
        """Check if there are any collisions between the robot and the environment
        Returns: Bool
        """
        collisions_success = self.pbclient.getContactPoints(
            bodyA=self.robot, bodyB=body_b, linkIndexA=self.success_link_index
        )
        for i in range(len(collisions_success)):
            if collisions_success[i][-6] < 0.05:
                if self.verbose > 1:
                    print("DEBUG: Success Collision")
                return True

        return False

    def set_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.pbclient.setCollisionFilterPair(self.robot, i, j, 0, 0)
        return

    def unset_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.pbclient.setCollisionFilterPair(self.robot, i, j, 0, 1)
        return

    # def get_camera_location(
    #     self, camera: Camera
    # ):  # TODO: get transform from dictionary. choose between rgb or tof frames
    #     pose, orientation = self.get_current_pose(camera.tf_id)
    #     tilt = camera.tilt

    #     camera_tf = self.create_camera_transform(camera=camera)
    #     return camera_tf

    # Collision checking
    #
    def deproject_pixels_to_points(
        self,
        sensor,
        data: np.ndarray,
        view_matrix: np.ndarray,
        return_frame: str = "world",
        debug=False,
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
        # log.debug(f"Data\n{data}")

        # Flip the y and z axes to convert from OpenGL camera frame to standard camera frame.
        # https://stackoverflow.com/questions/4124041/is-opengl-coordinate-system-left-handed-or-right-handed
        # https://github.com/bitlw/LearnProjMatrix/blob/main/doc/OpenGL_Projection.md#introduction
        # view_matrix[1:3, :] = -view_matrix[1:3, :]
        #

        proj_matrix = np.asarray(sensor.depth_proj_mat).reshape([4, 4], order="F")
        # log.warning(f'{proj_matrix}')
        # proj_matrix = camera.depth_proj_mat

        # rgb, depth = self.pbutils.get_rgbd_at_cur_pose(type='robot', view_matrix=view_matrix)
        # data = depth.reshape((self.cam_width * self.cam_height, 1), order="F")

        # Get camera intrinsics from projection matrix. If square camera, these should be the same.
        fx = proj_matrix[0, 0]
        fy = proj_matrix[1, 1]  # if square camera, these should be the same

        # Get camera coordinates from film-plane coordinates. Scale, add z (depth), then homogenize the matrix.
        sensor_coords = np.divide(np.multiply(sensor.depth_film_coords, data), [fx, fy])
        sensor_coords = np.concatenate(
            (
                sensor_coords,
                data,
                np.ones((sensor.depth_width * sensor.depth_height, 1)),
            ),
            axis=1,
        )

        return_frame = return_frame.strip().lower()
        if return_frame == "sensor":
            return sensor_coords
        elif return_frame == "world":
            world_coords = (mr.TransInv(view_matrix) @ sensor_coords.T).T
            # log.err("HELLO WORLD")
            if debug:
                plot.debug_deproject_pixels_to_points(
                    sensor=sensor,
                    data=data,
                    cam_coords=sensor_coords,
                    world_coords=world_coords,
                    view_matrix=view_matrix,
                )
            return world_coords
        else:
            raise ValueError("Invalid return frame. Must be 'camera' or 'world'.")

    def get_cam_to_frame_coords(
        self,
        cam_coords: np.ndarray,
        start_frame: str,
        end_frame: str = "world",
        view_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert camera coordinates to other frame coordinates. Default is world.
        @param cam_coords: nx4 array of camera XYZ coordinates
        @param start_frame: str
        @param end_frame: str (default = 'world')
        @param view_matrix: 4x4 matrix (frame -> camera transform)

        @return: nx4 array of world XYZ coordinates
        """
        end_frame = end_frame.strip().lower()
        if end_frame == "world" and view_matrix is None:
            raise ValueError("View matrix required for world frame conversion.")
        elif end_frame == "world":
            return (mr.TransInv(view_matrix) @ cam_coords.T).T

        start_frame = start_frame.strip().lower()
        end_frame_coords = (mr.TransInv(self.static_frames[f"{start_frame}_to_{end_frame}"]) @ cam_coords.T).T

        return end_frame_coords

    def compute_deprojected_point_mask(self):
        point_mask = []
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

    def get_key_move_action(self, keys_pressed: list) -> np.ndarray:
        """Return an action based on the keys pressed."""
        action = np.zeros((6, 1), dtype=float)
        if keys_pressed:
            if ord("a") in keys_pressed:
                action[0, 0] += 0.01
            if ord("d") in keys_pressed:
                action[0, 0] += -0.01
            if ord("s") in keys_pressed:
                action[1, 0] += 0.01
            if ord("w") in keys_pressed:
                action[1, 0] += -0.01
            if ord("q") in keys_pressed:
                action[2, 0] += 0.01
            if ord("e") in keys_pressed:
                action[2, 0] += -0.01
            if ord("z") in keys_pressed:
                action[3, 0] += 0.01
            if ord("c") in keys_pressed:
                action[3, 0] += -0.01
            if ord("x") in keys_pressed:
                action[4, 0] += 0.01
            if ord("v") in keys_pressed:
                action[4, 0] += -0.01
            if ord("r") in keys_pressed:
                action[5, 0] += 0.05
            if ord("f") in keys_pressed:
                action[5, 0] += -0.05
        return action

    def get_key_sensor_action(self, keys_pressed: list) -> dict | None:
        if keys_pressed:
            if ord("p") in keys_pressed:
                if time.time() - self.debounce_time > 0.1:
                    sensor_data = {}
                    for sensor_name, sensor in self.sensors.items():
                        if sensor_name.startswith("tof"):
                            view_matrix = self.get_view_mat_at_curr_pose(camera=sensor)
                            rgb, depth = self.get_rgbd_at_cur_pose(
                                camera=sensor,
                                type="sensor",
                                view_matrix=view_matrix,
                            )
                            view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
                            depth = depth.reshape(
                                (sensor.depth_width * sensor.depth_height, 1),
                                order="F",
                            )

                            camera_points = self.deproject_pixels_to_points(
                                sensor=sensor,
                                data=depth,
                                view_matrix=view_matrix,
                                return_frame="world",
                                debug=False,
                            )

                            sensor_data.update(
                                {
                                    sensor_name: {
                                        "data": camera_points,
                                        "tf_frame": sensor.tf_frame,
                                        "view_matrix": view_matrix,
                                        "sensor": sensor,
                                    }
                                }
                            )
                    # plot.debug_sensor_world_data(sensor_data)
                    self.debounce_time = time.time()
                    return sensor_data
                else:
                    return
        return

    def get_key_action(self, keys_pressed: list):
        move_action = self.get_key_move_action(keys_pressed=keys_pressed)
        sensor_data = self.get_key_sensor_action(keys_pressed=keys_pressed)
        # controller_action = self.get_key_controller_action(keys_pressed=keys_pressed)

        return


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import time

    pbutils = PyBUtils(renders=True)

    robot = Robot(
        pbclient=pbutils.pbclient,
    )

    return


if __name__ == "__main__":
    main()
