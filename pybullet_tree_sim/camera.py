#!/usr/bin/env python3
"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.sensor import Sensor

import numpy as np
import os

from zenlog import log


# class Camera(Sensor):
#     def __init__(self, pbutils: PyBUtils, cam_width: int, cam_height: int, sensor_name: str, sensor_type: str='camera') -> None:
#         """Builds a camera object from a base Sensor class"""
#         super().__init__(sensor_name=sensor_name, sensor_type=sensor_type)
#         self.cam_width = cam_width
#         self.cam_height = cam_height
        
#         self.pixel_coords = np.array(list(np.ndindex((cam_width, cam_height))), dtype=int)
#         self.film_coords = 2 * np.divide(
#             np.subtract(np.add(self.pixel_coords, [0.5, 0.5]), [cam_width / 2, cam_height / 2]), [cam_width, cam_height]
#         )
#         self.proj_mat = pbutils.pbclient.computeProjectionMatrixFOV(
#             fov=vfov, aspect=cam_width / cam_height, nearVal=0.01, farVal=100
#         )
        
#         return   
# 
class Camera(Sensor):
    def __init__(self, pbutils: PyBUtils, sensor_name: str, sensor_type: str = 'camera') -> None:
        super().__init__(sensor_name, sensor_type)
        
        return


def main():
    pbutils = PyBUtils(renders=False, cam_width=8, cam_height=8, dfov=65)
    camera = Camera(pbutils, sensor_name="realsense_d435i")
    print(camera.params)
    return


if __name__ == "__main__":
    main()
