from typing import Optional, Tuple
import numpy as np
import pybullet
from collections import namedtuple

class Robot:
    def __init__(
        self, 
        con, 
        robot_type: str, 
        robot_urdf_path: str,
        tool_link_name: str,
        end_effector_link_name: str,
        success_link_name: str,
        base_link_name: str,
        control_joints,
        robot_collision_filter_idxs,
        init_joint_angles: Optional[list] = None,
        pos=(0, 0, 0), 
        orientation=(0, 0, 0, 1), 
        randomize_pose=False, 
        verbose=1
    ) -> None:
        self.con = con
        self.robot_type = robot_type
        self.robot_urdf_path = robot_urdf_path
        self.tool_link_name = tool_link_name
        self.end_effector_link_name = end_effector_link_name
        self.success_link_name = success_link_name
        self.base_link_name = base_link_name
        self.init_joint_angles = init_joint_angles
        self.pos = pos
        self.orientation = orientation
        self.randomize_pose = randomize_pose

        self.joint_info = namedtuple(
            "jointInfo", 
            ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"]
        )
        # Pruning camera information
        self.camera_base_offset = np.array(
            [0.063179, 0.077119, 0.0420027])
        self.verbose = verbose
        
        self.joints = None
        self.robot = None
        self.control_joints = control_joints
        self.robot_collision_filter_idxs = robot_collision_filter_idxs
        self.setup_robot()

    def setup_robot(self):
        if self.robot is not None:
            self.con.removeBody(self.robot)
            del self.robot
        flags = self.con.URDF_USE_SELF_COLLISION

        if self.randomize_pose:
            delta_pos = np.random.rand(3) * 0.0
            delta_orientation = pybullet.getQuaternionFromEuler(np.random.rand(3) * np.pi / 180 * 5)
        else:
            delta_pos = np.array([0., 0., 0.])
            delta_orientation = pybullet.getQuaternionFromEuler([0, 0, 0])

        self.pos, self.orientation = self.con.multiplyTransforms(
            self.pos, 
            self.orientation, 
            delta_pos, 
            delta_orientation
        )
        self.robot = self.con.loadURDF(
            self.robot_urdf_path, 
            self.pos, 
            self.orientation, 
            flags=flags, 
            useFixedBase=True
        )
        self.num_joints = self.con.getNumJoints(self.robot)

        #Get indices dynamically
        self.tool0_link_index = self.get_link_index(self.tool_link_name)
        self.end_effector_index = self.get_link_index(self.end_effector_link_name)
        self.success_link_index = self.get_link_index(self.success_link_name)
        self.base_index = self.get_link_index(self.base_link_name)

        self.set_collision_filter()

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
                info = self.con.getJointInfo(self.robot, i)
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

                if info.type == self.con.JOINT_REVOLUTE:
                    self.con.setJointMotorControl2(
                        self.robot,
                        info.id,
                        self.con.VELOCITY_CONTROL,
                        targetVelocity=0,
                        force=0,
                    )
                self.joints[info.name] = info

        self.set_joint_angles_no_collision(self.init_joint_angles)
        self.con.stepSimulation()

        self.init_pos_ee = self.get_current_pose(self.end_effector_index)
        self.init_pos_base = self.get_current_pose(self.base_index)
        self.init_pos_eebase = self.get_current_pose(self.success_link_index)
        self.action = np.zeros(len(self.init_joint_angles), dtype=np.float32)
        self.joint_angles = np.array(self.init_joint_angles).astype(np.float32)
        self.achieved_pos = np.array(self.get_current_pose(self.end_effector_index)[0])
        base_pos, base_or = self.get_current_pose(self.base_index)

    def get_link_index(self, link_name):
        num_joints = self.con.getNumJoints(self.robot)
        for i in range(num_joints):
            info = self.con.getJointInfo(self.robot, i)
            child_link_name = info[12].decode('utf-8')
            if child_link_name == link_name:
                return i # return link index
        
        base_link_name = self.con.getBodyInfo(self.robot)[0].decode('utf-8')
        if base_link_name == link_name:
            return -1 #base link has index of -1
        raise ValueError(f"Link '{link_name}' not found in the robot URDF.")
    
    def reset_robot(self):
        if self.robot is None:
            return

        self.set_joint_angles_no_collision(self.init_joint_angles)

    def remove_robot(self):
        self.con.removeBody(self.robot)
        self.robot = None

    def set_joint_angles_no_collision(self, joint_angles) -> None:
        assert len(joint_angles) == len(self.control_joints)
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            self.con.resetJointState(self.robot, joint.id, joint_angles[i], targetVelocity=0)

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

        self.con.setJointMotorControlArray(
            self.robot, indexes,
            self.con.POSITION_CONTROL,
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

        self.con.setJointMotorControlArray(self.robot,
                                           indexes,
                                           controlMode=self.con.VELOCITY_CONTROL,
                                           targetVelocities=joint_velocities,
                                           )

    # TODO: Use proprty decorator for getters?
    def get_joint_velocities(self):
        j = self.con.getJointStates(self.robot, self.controllable_joints_idxs)
        joints = tuple((i[1] for i in j))
        return joints  # type: ignore

    def get_joint_angles(self):
        """Return joint angles"""
        print(self.control_joints, self.controllable_joints_idxs)
        j = self.con.getJointStates(self.robot, self.controllable_joints_idxs)
        joints = tuple((i[0] for i in j))
        return joints

    def get_current_pose(self, index):
        """Returns current pose of the index"""
        link_state = self.con.getLinkState(self.robot, index, computeForwardKinematics=True)
        position, orientation = link_state[4], link_state[5]
        return position, orientation

    def get_current_vel(self, index):
        """Returns current pose of the index."""
        link_state = self.con.getLinkState(self.robot, index, computeLinkVelocity=True,
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

        joint_angles = self.con.calculateInverseKinematics(
            self.robot, self.end_effector_index, position, orientation,
            jointDamping=[0.01] * len(self.control_joints), upperLimits=self.joint_upper_limits,
            lowerLimits=self.joint_lower_limits, jointRanges=self.joint_ranges  # , restPoses=self.init_joint_angles
        )
        return joint_angles

    def calculate_jacobian(self):
        jacobian = self.con.calculateJacobian(self.robot, self.tool0_link_index, [0, 0, 0],
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
    def create_camera_transform(self, pos, orientation, pan, tilt, xyz_offset) -> np.ndarray:
        """Create rotation matrix for camera"""
        base_offset_tf = np.identity(4)
        base_offset_tf[:3, 3] = self.camera_base_offset + xyz_offset

        ee_transform = np.identity(4)
        ee_rot_mat = np.array(self.con.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        ee_transform[:3, :3] = ee_rot_mat
        ee_transform[:3, 3] = pos

        tilt_tf = np.identity(4)
        tilt_rot = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        tilt_tf[:3, :3] = tilt_rot

        pan_tf = np.identity(4)
        pan_rot = np.array([[np.cos(pan), 0, np.sin(pan)], [0, 1, 0], [-np.sin(pan), 0, np.cos(pan)]])
        pan_tf[:3, :3] = pan_rot

        tf = ee_transform @ pan_tf @ tilt_tf @ base_offset_tf
        return tf

    # TODO: Better types for getCameraImage
    def get_view_mat_at_curr_pose(self, pan, tilt, xyz_offset) -> np.ndarray:
        """Get view matrix at current pose"""
        pose, orientation = self.get_current_pose(self.tool0_link_index)

        camera_tf = self.create_camera_transform(pose, orientation, pan, tilt, xyz_offset)

        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T  #
        up_vector = np.array([0, 1, 0]) @ camera_tf[:3, :3].T  #
        # Rotated vectors
        # print(camera_vector, up_vector)
        view_matrix = self.con.computeViewMatrix(camera_tf[:3, 3], camera_tf[:3, 3] + 0.1 * camera_vector, up_vector)
        return view_matrix

    def get_camera_location(self):
        pose, orientation = self.get_current_pose(self.tool0_link_index)
        tilt = np.pi / 180 * 8

        camera_tf = self.create_camera_transform(pose, orientation, tilt)
        return camera_tf

    # Collision checking

    def set_collision_filter(self):
        """Disable collision between pruner and arm"""
        for i in self.robot_collision_filter_idxs:
            self.con.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 0)

    def unset_collision_filter(self):
        """Enable collision between pruner and arm"""
        for i in self.robot_collision_filter_idxs:
            self.con.setCollisionFilterPair(self.robot, self.robot, i[0], i[1], 1)
    def disable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.con.setCollisionFilterPair(self.robot, self.robot, i, j, 0)

    def enable_self_collision(self):
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.con.setCollisionFilterPair(self.robot, self.robot, i, j, 1)

    def check_collisions(self, collision_objects) -> Tuple[bool, dict]:
        """Check if there are any collisions between the robot and the environment
        Returns: Dictionary with information about collisions (Acceptable and Unacceptable)
        """
        collision_info = {"collisions_acceptable": False, "collisions_unacceptable": False}

        collision_acceptable_list = ['SPUR', 'WATER_BRANCH']
        collision_unacceptable_list = ['TRUNK', 'BRANCH', 'SUPPORT']
        for type in collision_acceptable_list:
            collisions_acceptable = self.con.getContactPoints(bodyA=self.robot, bodyB=collision_objects[type])
            if collisions_acceptable:
                for i in range(len(collisions_acceptable)):
                    if collisions_acceptable[i][-6] < 0:
                        collision_info["collisions_acceptable"] = True
                        break
            if collision_info["collisions_acceptable"]:
                break

        for type in collision_unacceptable_list:
            collisions_unacceptable = self.con.getContactPoints(bodyA=self.robot, bodyB=collision_objects[type])
            for i in range(len(collisions_unacceptable)):
                if collisions_unacceptable[i][-6] < 0:
                    collision_info["collisions_unacceptable"] = True
                    # break
            if collision_info["collisions_unacceptable"]:
                break

        if not collision_info["collisions_unacceptable"]:
            collisons_self = self.con.getContactPoints(bodyA=self.robot, bodyB=self.robot)
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
        collisions_success = self.con.getContactPoints(bodyA=self.robot, bodyB=body_b,
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
                self.con.setCollisionFilterPair(self.robot, i, j, 0, 0)

    def unset_collision_filter_tree(self, collision_objects):
        for i in collision_objects.values():
            for j in range(self.num_joints):
                self.con.setCollisionFilterPair(self.robot, i, j, 0, 1)