#!/usr/bin/env python3
from __future__ import annotations

"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.utils import camera_helpers as ch
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.sensor import Sensor

import numpy as np
import os

from zenlog import log


class Camera(Sensor):
    def __init__(self, pbutils, sensor_name: str, sensor_type: str = "camera") -> None:
        # load sensor parameters from yaml file
        super().__init__(sensor_name, sensor_type)
        self.pan = 0
        self.tilt = 0

        # Only dealing with depth data for now, TODO: add RGB data
        self.depth_width = self.params["depth"]["width"]
        self.depth_height = self.params["depth"]["height"]
        
        # Some optical sensors only provide diagonal field of view, get horizontal and vertical from diagonal
        try:
            vfov = self.params["depth"]["vfov"]
        except KeyError:
            vfov = ch.get_fov_from_dfov(self.params["depth"]["dfov"], self.depth_width, self.depth_height)[1]
        near_val = self.params["depth"]["near_plane"]
        far_val = self.params["depth"]["far_plane"]

        self.depth_pixel_coords = np.array(list(np.ndindex((self.depth_width, self.depth_height))), dtype=int)
        self.depth_film_coords = 2 * (
            self.depth_pixel_coords + np.array([0.5, 0.5]) - np.array([self.depth_width / 2, self.depth_height / 2])
        ) / np.array([self.depth_width, self.depth_height])
        self.depth_proj_mat = pbutils.pbclient.computeProjectionMatrixFOV(
            fov=vfov, aspect=(self.depth_width / self.depth_height), nearVal=near_val, farVal=far_val
        )
        return


def main():
    pbutils = PyBUtils(renders=False)
    camera = Camera(pbutils, sensor_name="realsense_d435i")
    print(camera.depth_film_coords)
    return


if __name__ == "__main__":
    main()
