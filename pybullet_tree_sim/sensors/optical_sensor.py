#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot

import logging
from abc import abstractmethod

import numpy as np
from pybullet_utils import bullet_client as bc

import pybullet_tree_sim.utils.logging_conf
from pybullet_tree_sim.sensors.sensor import Sensor
from pybullet_tree_sim.sensors.sensor_types import (
    DataType,
    Modality,
    data_type_modalities,
)
from pybullet_tree_sim.utils import camera_helpers as ch
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OpticalSensor(Sensor):
    """Base class for sensors with camera intrinsics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__pan = 0.0
        self.__tilt = 0.0  # TODO: these are fillers, original settings are in YAML. Remove and just do URDF or keep?

        # Upsample image for low-resolution sensors to get accurate pixel values. This avoids color smearing.
        self.upsample = self.params.get("upsample", False)
        self.upsample_factor = self.params.get("upsample_factor", 9)

        self.__intrinsics = {}
        for mode in self.modalities:
            mode_params = self.params.get(mode, {})
            # Calculate focal lengths from FOV
            hfov = mode_params.get("hfov", None)
            vfov = mode_params.get("vfov", None)
            if hfov is None or vfov is None:
                dfov = mode_params.get("dfov", None)
                hfov, vfov = ch.get_fov_from_dfov(mode_params["width"], mode_params["height"], dfov)
            fx = mode_params["width"] / (2 * np.tan(np.radians(hfov) / 2))
            fy = mode_params["height"] / (2 * np.tan(np.radians(vfov) / 2))

            # Principal point at image center
            cx = mode_params["width"] / 2
            cy = mode_params["height"] / 2

            self.__intrinsics[mode] = dict(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=mode_params["width"],
                height=mode_params["height"],
                znear=mode_params["z_near"],
                zfar=mode_params["z_far"],
            )
        return

    @property
    def optical_intrinsics(self):
        """Convert sensor parameters to standard camera intrinsics.
        :return: Dict with camera intrinsics. Keys -- mode (sensor_types.Modality): dict with keys fx, fy, cx, cy, width, height, znear, zfar
        :rtype: dict
        """
        return self.__intrinsics

    def read(self, robot: Robot, pbclient: bc.BulletClient, mode: Modality) -> dict:
        """Abstract method to read data from the sensor. # TODO: fix docstrings
        :param data_type: The data type to read from the sensor.
        :type data_type: DataType
        :param upsample: Whether to upsample the image.
        :type upsample: bool
        :return: The data read from the sensor.
        :rtype: np.ndarray
        """
        if mode not in self.modalities:
            allowed = ", ".join(m.value for m in self.modalities)
            raise ValueError(f"Unsupported mode '{mode}' for sensor '{self.sensor_name}'. Allowed: {allowed}")
        results_dict = {}
        view_mat = self.get_view_mat_at_curr_pose(robot=robot, pbclient=pbclient)
        rgbd_img = self.get_rgbd_at_cur_pose(
            view_type="sensor",
            data_mode=mode,
            view_matrix=view_mat,
            pbclient=pbclient,
        )
        results_dict["data"] = rgbd_img
        results_dict["extrinsic_matrix"] = view_mat
        results_dict["mode"] = mode
        results_dict["intrinsics"] = self.optical_intrinsics[mode]
        return results_dict

    def create_sensor_transform(self, world_position, world_orientation, pbclient: bc.BulletClient) -> np.ndarray:
        """Create rotation matrix for camera"""
        base_offset_tf = np.identity(4)

        ee_transform = np.identity(4)
        ee_rot_mat = np.array(pbclient.getMatrixFromQuaternion(world_orientation)).reshape(3, 3)

        ee_transform[:3, :3] = ee_rot_mat
        ee_transform[:3, 3] = world_position

        tilt_tf = np.identity(4)
        pan_tf = np.identity(4)
        tilt = self.__tilt
        pan = self.__pan
        base_offset_tf[:3, 3] = self.xyz_offset

        tilt_rot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        tilt_tf[:3, :3] = tilt_rot

        pan_rot = np.array(
            [
                [np.cos(pan), 0, np.sin(pan)],
                [0, 1, 0],
                [-np.sin(pan), 0, np.cos(pan)],
            ]
        )
        pan_tf[:3, :3] = pan_rot

        tf = ee_transform @ pan_tf @ tilt_tf @ base_offset_tf
        return tf

    def get_view_mat_at_curr_pose(self, robot: Robot, pbclient: bc.BulletClient) -> np.ndarray:
        """Get view matrix at current pose"""
        pos, orientation = robot.get_current_pose(self.tf_id)
        camera_tf = self.create_sensor_transform(pos, orientation, pbclient)

        # Initial vectors
        camera_vector = np.array([0, 0, 1]) @ camera_tf[:3, :3].T  #
        up_vector = np.array([0, -1, 0]) @ camera_tf[:3, :3].T  #

        view_matrix = np.asarray(
            pbclient.computeViewMatrix(
                cameraEyePosition=camera_tf[:3, 3],
                cameraTargetPosition=camera_tf[:3, 3] + 0.1 * camera_vector,
                cameraUpVector=up_vector,
            )
        )
        return view_matrix

    def get_view_mat_by_id_at_curr_pose(self, robot: Robot, pbclient: bc.BulletClient) -> np.ndarray:
        pos, orientation = robot.get_current_pose(index=self.tf_id)
        sensor_tf = self.create_sensor_transform(world_position=pos, world_orientation=orientation, pbclient=pbclient)
        # Initial vectors
        view_vector = np.array([0, 0, 1]) @ sensor_tf[:3, :3].T
        up_vector = np.array([0, -1, 0]) @ sensor_tf[:3, :3].T

        view_matrix = pbclient.computeViewMatrix(
            cameraEyePosition=sensor_tf[:3, 3],
            cameraTargetPosition=sensor_tf[:3, 3] + 0.1 * view_vector,
            cameraUpVector=up_vector,
        )
        return view_matrix

    def get_rgbd_at_cur_pose(
        self, view_type: str, data_mode: Modality, view_matrix: np.ndarray, pbclient: bc.BulletClient
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get RGBD image at current pose

        :param view_type: either 'robot' or 'viz'
        :param data_mode: A value from `Modality` enum specifying the type of data to capture (e.g., 'rgb', 'depth', 'rgbd')
        :param view_matrix: 16x1 tuple representing the view matrix
        :param pbclient: PyBullet client object

        :return: (rgb, depth) tuple of RGB and depth images
        """
        rgbd = self.get_image_at_curr_pose(
            view_type=view_type, data_mode=data_mode, pbclient=pbclient, view_matrix=view_matrix
        )
        if data_mode == Modality.RGB:
            width = self.rgb_width
            height = self.rgb_height
        elif data_mode == Modality.DEPTH:
            width = self.depth_width
            height = self.depth_height
        else:
            raise ValueError(f"Unsupported data_mode {data_mode} for RGBD capture. Must be 'rgb' or 'depth'.")
            
        if self.upsample:
            width *= self.upsample_factor
            height *= self.upsample_factor

        rgb, depth = ch.seperate_rgbd_rgb_d(rgbd=rgbd, width=int(width), height=int(height))
        depth = depth.astype(np.float32)
        depth = PyBUtils.linearize_depth(
            depth=depth,
            near_val=self.z_near,
            far_val=self.z_far
        )

        # downsample if upsample is true
        if self.upsample:
            rgb, depth = ch.downsample_rgbd(rgb, depth, width, height, self.upsample_factor)

        return rgb, depth

    def get_image_at_curr_pose(
        self, view_type: str, data_mode: Modality, pbclient: bc.BulletClient, view_matrix=None
    ) -> list:
        """Take the current pose of the sensor and capture an image
        TODO: Add support for different types of sensors? For now, full rgbd
        TOOD: Move sensor/viz view to different methods, viz to pruning env?"""
        if data_mode == Modality.RGB:
            width = self.rgb_width
            height = self.rgb_height
            proj_mat = self.rgb_proj_mat
        elif data_mode == Modality.DEPTH:
            width = self.depth_width
            height = self.depth_height
            proj_mat = self.depth_proj_mat
        else:
            raise ValueError(f"data_mode {data_mode} not recognized, must be 'rgb' or 'depth'")

        if self.upsample:
            width *= self.upsample_factor
            height *= self.upsample_factor

        if view_type == "sensor":
            if view_matrix is None:
                raise ValueError("view_matrix cannot be None for sensor view")
            return pbclient.getCameraImage(
                width=int(width),  # TODO: how to work with depth + RGB?
                height=int(height),
                viewMatrix=view_matrix,
                projectionMatrix=proj_mat,  # TODO: ^ same
                renderer=pbclient.ER_BULLET_HARDWARE_OPENGL,
                flags=pbclient.ER_NO_SEGMENTATION_MASK,
                lightDirection=[1, 1, 1],
            )
        elif view_type == "viz":
            return pbclient.getCameraImage(
                width=int(width),
                height=int(height),
                viewMatrix=self.viz_view_matrix,
                projectionMatrix=self.viz_proj_matrix,
                renderer=pbclient.ER_BULLET_HARDWARE_OPENGL,
                flags=pbclient.ER_NO_SEGMENTATION_MASK,
                lightDirection=[1, 1, 1],
            )


def main():
    import pprint as pp

    os_realsense_d435i = OpticalSensor(
        sensor_model="realsense_d435i", sensor_type="depth_camera", sensor_name="test_depth_sensor"
    )
    logger.debug(pp.pformat(os_realsense_d435i.optical_intrinsics))

    os_vl53l8cx = OpticalSensor(sensor_model="vl53l8cx", sensor_type="tof", sensor_name="test_tof_sensor")
    logger.debug(pp.pformat(os_vl53l8cx.optical_intrinsics))
    return


if __name__ == "__main__":
    main()
