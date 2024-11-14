#!/usr/bin/env python3
import pybullet_tree_sim.utils.xacro_utils as xutils
import pybullet_tree_sim.utils.yaml_utils as yutils
from pybullet_tree_sim import CONFIG_PATH, URDF_PATH

from typing import Optional, Tuple
import numpy as np
import pybullet
from collections import namedtuple
import os

from zenlog import log

class Robot:

    _robot_configs_path = os.path.join(CONFIG_PATH, "robot")
    _robot_xacro_path = os.path.join(URDF_PATH, "robot", "generic", "robot.urdf.xacro")
    _urdf_tmp_path = os.path.join(URDF_PATH, "tmp")

    def __init__(
        self,
        pbclient,
        # robot_type: str,
        # robot_urdf_path: str,
        # tool_link_name: str,
        # end_effector_link_name: str,
        # success_link_name: str,
        # base_link_name: str,
        # control_joints,
        # robot_collision_filter_idxs,
        # init_joint_angles: Optional[list] = None,
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        randomize_pose=False,
        verbose=1
        # **kwargs
    ) -> None:
        self.pbclient = pbclient
        # self.robot_type = robot_type
        # self.robot_urdf_path = robot_urdf_path
        # self.tool_link_name = tool_link_name
        # self.end_effector_link_name = end_effector_link_name
        # self.success_link_name = success_link_name
        # self.base_link_name = base_link_name
        # self.init_joint_angles = init_joint_angles
        self.pos = position
        self.orientation = orientation
        self.randomize_pose = randomize_pose

        # self.joint_info = namedtuple(
        #     "jointInfo",
        #     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"]
        # )
        # # Pruning camera information
        # self.camera_base_offset = np.array(
        #     [0.063179, 0.077119, 0.0420027])
        self.verbose = verbose

        self.joints = None
        self.robot = None
        # self.control_joints = control_joints
        # self.robot_collision_filter_idxs = robot_collision_filter_idxs
        
        
        self.robot_conf = {}
        self._generate_robot_urdf()
        self.setup_robot()

        return

    def _generate_robot_urdf(self) -> None:
        # Get robot params
        self.robot_conf.update(yutils.load_yaml(os.path.join(self._robot_configs_path, "robot.yaml")))
        # Add the required urdf args from each element of the robot_stack config
        for robot_part in self.robot_conf["robot_stack"]:
            robot_part = robot_part.strip().lower()
            self.robot_conf.update(yutils.load_yaml(os.path.join(self._robot_configs_path, f"{robot_part}.yaml")))
        
        # Generate URDF from mappings
        robot_urdf = xutils.load_urdf_from_xacro(
            xacro_path=self._robot_xacro_path,
            mappings=self.robot_conf # for some reason, this adds in the rest of the args from the xacro.
        )

        # UR_description uses filename="package://<>" for meshes, and this doesn't work with pybullet
        if self.robot_conf['arm_type'].startswith('ur'):
            ur_absolute_mesh_path = '/opt/ros/humble/share/ur_description/meshes'
            robot_urdf = robot_urdf.toprettyxml().replace(
                f'filename="package://ur_description/meshes',
                f'filename="{ur_absolute_mesh_path}'
            )
        else:
            robot_urdf = robot_urdf.toprettyxml()
            
        # Save the generated URDF
        self.robot_urdf_path = os.path.join(self._urdf_tmp_path, "robot.urdf")
        xutils.save_urdf(robot_urdf, urdf_path=self.robot_urdf_path)
        return

    def setup_robot(self):
        if self.robot is not None:
            self.pbclient.removeBody(self.robot)
            del self.robot
        flags = self.pbclient.URDF_USE_SELF_COLLISION

        if self.randomize_pose:
            delta_pos = np.random.rand(3) * 0.0
            delta_orientation = pybullet.getQuaternionFromEuler(np.random.rand(3) * np.pi / 180 * 5)
        else:
            delta_pos = np.array([0., 0., 0.])
            delta_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])

        self.pos, self.orientation = self.pbclient.multiplyTransforms(
            self.pos,
            self.orientation,
            delta_pos,
            delta_orientation
        )
        self.robot = self.pbclient.loadURDF(
            self.robot_urdf_path,
            self.pos,
            self.orientation,
            flags=flags,
            useFixedBase=True
        )
        self.num_joints = self.pbclient.getNumJoints(self.robot)
        
        # get link indices dynamically
        self._assign_link_indices()
        
        self.set_collision_filter()
        
        import pprint as pp
        pp.pprint(self.robot_conf)

        #Setup robot info only once
        if not self.joints:
            self.joints = dict()
            self.controllable_joints_idxs = []
            self.joint_lower_limits = []
            self.joint_upper_limits = []
            self.joint_max_forces = []
            self.joint_max_velocities = []
            self.joint_ranges = []

            for i in range(self.num_joints):
                info = self.pbclient.getJointInfo(self.robot, i)
                jointID = info[0]
                jointName = info[1].decode("utf-8")
                jointType = info[2]
                jointLowerLimit = info[8]
                jointUpperLimit = info[9]
                jointMaxForce = info[10]
                jointMaxVelocity = info[11]
                if self.verbose > 1:
                    print("Joint Name: ", jointName, "Joint ID: ", jointID)

                controllable = True if jointName in self.control_joints else False
                if controllable:
                    self.controllable_joints_idxs.append(i)
                    self.joint_lower_limits.append(jointLowerLimit)
                    self.joint_upper_limits.append(jointUpperLimit)
                    self.joint_max_forces.append(jointMaxForce)
                    self.joint_max_velocities.append(jointMaxVelocity)
                    self.joint_ranges.append(jointUpperLimit - jointLowerLimit)
                    if self.verbose > 1:
                        print("Controllable Joint Name: ", jointName, "Joint ID: ", jointID)

                info = self.joint_info(
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable
                )

                if info.type == self.pbclient.JOINT_REVOLUTE:
                    self.pbclient.setJointMotorControl2(
                        self.robot,
                        info.id,
                        self.pbclient.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=0,
                    )
                self.joints[info.name] = info

        self.set_joint_angles_no_collision(self.init_joint_angles)
        self.pbclient.stepSimulation()

        # self.init_pos_ee = self.get_current_pose(self.end_effector_index)
        # self.init_pos_base = self.get_current_pose(self.base_index)
        # self.init_pos_eebase = self.get_current_pose(self.success_link_index)
        # self.action = np.zeros(len(self.init_joint_angles), dtype=np.float32)
        self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        # self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        # base_pos, base_or = self.get_current_pose(self.base_index)
        return

    # def get_link_index(self, link_name):
    #     num_joints = self.pbclient.getNumJoints(self.robot)
    #     for i in range(num_joints):
    #         info = self.pbclient.getJointInfo(self.robot, i)
    #         child_link_name = info[12].decode('utf-8')
    #         log.warn(child_link_name)
    #         self.robot_conf["tf_frames"].update({child_link_name: i})

    #         if child_link_name == link_name:
    #             return i # return link index

    #     base_link_name = self.pbclient.getBodyInfo(self.robot)[0].decode('utf-8')
    #     if base_link_name == link_name:
    #         return -1 #base link has index of -1
    #     raise ValueError(f"Link '{link_name}' not found in the robot URDF.")
    
    def _assign_link_indices(self):
        self.robot_collision_filter_idxs = []
        num_joints = self.pbclient.getNumJoints(self.robot)
        self.robot_conf["joint_info"] = {}
        prev_link_parent = 'world'
        for i in range(num_joints):
            info = self.pbclient.getJointInfo(self.robot, i)
            child_link_name = info[12].decode('utf-8')
            
            # This is kinda hacky, but it works for now. TODO: make better?
            if child_link_name.endswith('base'):
                self.robot_collision_filter_idxs.append((i, i-1))
            self.robot_conf["joint_info"].update({child_link_name: i})
        log.warn(self.robot_collision_filter_idxs)
        return
        
    def reset_robot(self):
        if self.robot is None:
            return

        self.set_joint_angles_no_collision(self.init_joint_angles)

    def remove_robot(self):
        self.pbclient.removeBody(self.robot)
        self.robot = None

    def set_joint_angles_no_collision(self, joint_angles) -> None:
        assert len(joint_angles) == len(self.control_joints)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            self.pbclient.resetJointState(self.robot, joint.id, joint_angles[i], targetVelocity=0)

    def set_joint_angles(self, joint_angles) -> None:
        """Set joint angles using pybullet motor control"""

        assert len(joint_angles) == len(self.control_joints)
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        self.pbclient.setJointMotorControlArray(
            self.robot, indexes,
            self.pbclient.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.05] * len(poses),
            forces=forces
        )

    def set_joint_velocities(self, joint_velocities) -> None:
        """Set joint velocities using pybullet motor control"""
        assert len(joint_velocities) == len(self.control_joints)
        velocities = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            velocities.append(joint_velocities[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        self.pbclient.setJointMotorControlArray(self.robot,
                                           indexes,
                                           controlMode=self.pbclient.VELOCITY_CONTROL,
                                           targetVelocities=joint_velocities,
                                           )

    # TODO: Use proprty decorator for getters?
    def get_joint_velocities(self):
        j = self.pbclient.getJointStates(self.robot, self.controllable_joints_idxs)
        joints = tuple((i[1] for i in j))
        return joints  # type: ignore

    def get_joint_angles(self):
        """Return joint angles"""
        print(self.control_joints, self.controllable_joints_idxs)
        j = self.pbclient.getJointStates(self.robot, self.controllable_joints_idxs)
        joints = tuple((i[0] for i in j))
        return joints

    def get_current_pose(self, index):
        """Returns current pose of the index"""
        link_state = self.pbclient.getLinkState(self.robot, index, computeForwardKinematics=True)
        position, orientation = link_state[4], link_state[5]
        return position, orientation

    def get_current_vel(self, index):
        """Returns current pose of the index."""
        link_state = self.pbclient.getLinkState(self.robot, index, computeLinkVelocity=True,
                                           computeForwardKinematics=True)
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
            self.robot, self.end_effector_index, position, orientation,
            jointDamping=[0.01] * len(self.control_joints), upperLimits=self.joint_upper_limits,
            lowerLimits=self.joint_lower_limits, jointRanges=self.joint_ranges  # , restPoses=self.init_joint_angles
        )
        return joint_angles

    def calculate_jacobian(self):
        jacobian = self.pbclient.calculateJacobian(self.robot, self.tool0_link_index, [0, 0, 0],
                                              self.get_joint_angles(),
                                              [0]*len(self.control_joints), [0]*len(self.control_joints))
        jacobian = np.vstack(jacobian)
        return jacobian

    def calculate_joint_velocities_from_ee_velocity(self, end_effector_velocity):
        """Calculate joint velocities from end effector velocity using jacobian using least squares"""
        jacobian = self.calculate_jacobian()
        inv_jacobian = np.linalg.pinv(jacobian)
        joint_velocities = np.matmul(inv_jacobian, end_effector_velocity).astype(np.float32)
        return joint_velocities, jacobian

    def calculate_joint_velocities_from_ee_velocity_dls(self,
                                                        end_effector_velocity,
                                                        damping_factor: float = 0.05):
        """Calculate joint velocities from end effector velocity using damped least squares"""
        jacobian = self.calculate_jacobian()
        identity_matrix = np.eye(jacobian.shape[0])
        damped_matrix = jacobian @ jacobian.T + (damping_factor ** 2) * identity_matrix
        damped_matrix_inv = np.linalg.inv(damped_matrix)
        dls_inv_jacobian = jacobian.T @ damped_matrix_inv
        joint_velocities = dls_inv_jacobian @ end_effector_velocity
        return joint_velocities, jacobian

    # TODO: Make camera a separate class?
    def create_camera_transform(self, world_position, world_orientation, camera: "Camera") -> np.ndarray:
        """Create rotation matrix for camera"""
        base_offset_tf = np.identity(4)
        base_offset_tf[:3, 3] = camera.xyz_offset

        ee_transform = np.identity(4)
        ee_rot_mat = np.array(self.pbclient.getMatrixFromQuaternion(world_orientation)).reshape(3, 3)
        # log.debug(f"EE Rot Mat:\n{ee_rot_mat}")

        ee_transform[:3, :3] = ee_rot_mat
        ee_transform[:3, 3] = world_position

        tilt_tf = np.identity(4)
        tilt_rot = np.array([[1, 0, 0], [0, np.cos(camera.tilt), -np.sin(camera.tilt)], [0, np.sin(camera.tilt), np.cos(camera.tilt)]])
        tilt_tf[:3, :3] = tilt_rot

        pan_tf = np.identity(4)
        pan_rot = np.array([[np.cos(camera.pan), 0, np.sin(camera.pan)], [0, 1, 0], [-np.sin(camera.pan), 0, np.cos(camera.pan)]])
        pan_tf[:3, :3] = pan_rot

        tf = ee_transform @ pan_tf @ tilt_tf @ base_offset_tf
        return tf

    # TODO: Better types for getCameraImage
    def get_view_mat_at_curr_pose(self, camera: "Camera") -> np.ndarray:
        """Get view matrix at current pose"""
        pos, orientation = self.get_current_pose(camera.tf_frame_index)
        # log.debug(f"tool0 Pose: {pose}, Orientation: {Rotation.from_quat(orientation).as_euler('xyz')}")

        camera_tf = self.create_camera_transform(pos, orientation, camera)

        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T  #
        up_vector = np.array([0, 1, 0]) @ camera_tf[:3, :3].T  #

        # log.debug(f"cam vec, up vec:\n{camera_vector}, {up_vector}")

        view_matrix = self.pbclient.computeViewMatrix(camera_tf[:3, 3], camera_tf[:3, 3] + 0.1 * camera_vector, up_vector)
        return view_matrix

    def get_camera_location(self, camera: "Camera"):  # TODO: get transform from dictionary. choose between rgb or tof frames
        pose, orientation = self.get_current_pose(camera.tf_frame_index)
        tilt = np.pi / 180 * 8

        camera_tf = self.create_camera_transform(camera=camera)
        return camera_tf

    # Collision checking

    def set_collision_filter(self):
        """Disable collision between pruner and arm"""
        for i in self.robot_collision_filter_idxs:
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
        collision_info = {"collisions_acceptable": False, "collisions_unacceptable": False}

        collision_acceptable_list = ['SPUR', 'WATER_BRANCH']
        collision_unacceptable_list = ['TRUNK', 'BRANCH', 'SUPPORT']
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
                if collisions_unacceptable[i][-6] < -0.001:
                    collision_info["collisions_unacceptable"] = True
                    break
        if self.verbose > 1:
            print(f"DEBUG: {collision_info}")

        if collision_info["collisions_acceptable"] or collision_info["collisions_unacceptable"]:
            return True, collision_info

        return False, collision_info

    def check_success_collision(self, body_b) -> bool:
        """Check if there are any collisions between the robot and the environment
        Returns: Boolw
        """
        collisions_success = self.pbclient.getContactPoints(bodyA=self.robot, bodyB=body_b,
                                                       linkIndexA=self.success_link_index)
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

    def unset_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.pbclient.setCollisionFilterPair(self.robot, i, j, 0, 1)


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import time
    pbutils = PyBUtils(renders=False)

    robot = Robot(
        pbclient = pbutils.pbclient,
        # robot_type="ur5e",
        # base_link_type="linear_slider",
        # end_effector_type="mock_pruner",

    )


    time.sleep(10)

    return


if __name__ == "__main__":
    main()
