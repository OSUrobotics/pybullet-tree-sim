#!/usr/bin/env python3
"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.sensor import Sensor

import numpy as np
import os

from zenlog import log

class Camera(Sensor):
    def __init__(self, cam_width: int, cam_height: int, sensor_name: str, sensor_type: str='camera') -> None:
        """Builds a camera object from a base Sensor class"""
        super().__init__(sensor_name=sensor_name, sensor_type=sensor_type)
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        self.pixel_coords = np.array(list(np.ndindex((cam_width, cam_height))), dtype=int)
        self.film_coords = 2 * np.divide(
            np.subtract(np.add(self.pixel_coords, [0.5, 0.5]), [cam_width / 2, cam_height / 2]), [cam_width, cam_height]
        )
        return   


def main():
    camera = Camera(sensor_name="realsense_d435i", cam_width=640, cam_height=480)
    print(camera.film_coords)
    return


if __name__ == "__main__":
    main()
