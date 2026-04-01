#!/usr/bin/env python3
from __future__ import annotations

"""Provides basic functions for a simulated depth camera.
Resources:
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.sensors.sensor_types import DataType, Modality
from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.utils import camera_helpers as ch
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
from pybullet_tree_sim.sensors.depth_sensor import DepthSensor
from pybullet_tree_sim.sensors.rgb_camera import RGBCamera
from pybullet_utils import bullet_client as bc


import numpy as np
import os

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DepthCamera(RGBCamera, DepthSensor):
    def __init__(self, sensor_type: str = "depth_camera", *args, **kwargs) -> None:
        super().__init__(sensor_type=sensor_type, *args, **kwargs)

        

        # Pixel coordinates, indexed by depth_width, depth_height, nx1 array COLUMN MAJOR
        self.depth_pixel_coords = np.array(list(np.ndindex((self.depth_width, self.depth_height))), dtype=int)
        # Film coordinates projected to [-1, 1], nx1 array COLUMN MAJOR
        self.depth_film_coords = (
            2
            * (self.depth_pixel_coords + np.array([0.5, 0.5]) - np.array([self.depth_width / 2, self.depth_height / 2]))
            / np.array([self.depth_width, self.depth_height])
        )
        # Depth projection matrix from camera intrinsics
        self.depth_proj_mat = pbclient.computeProjectionMatrixFOV(
            fov=self.depth_vfov,
            aspect=(self.depth_width / self.depth_height),
            nearVal=self.near_val,
            farVal=self.far_val,
        )

        return

    def read(
        self, robot: Robot, pbclient: bc.BulletClient, return_view_mat: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read data from the depth camera sensor."""
        return super().read(robot=robot, pbclient=pbclient, return_view_mat=return_view_mat)


def main():
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    dcamera = DepthCamera(pbclient=pbutils.pbclient, sensor_model="realsense_d435i", sensor_name="test_depth_camera")
    print(DepthCamera.__mro__)
    print(dcamera.get_optical_intrinsics())
    pp.pprint(dcamera.params)
    pp.pprint(dcamera.depth_pixel_coords)
    pp.pprint(dcamera.depth_film_coords)
    pp.pprint(dcamera.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
