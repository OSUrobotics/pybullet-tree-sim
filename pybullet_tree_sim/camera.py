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
    def __init__(self, pbclient, sensor_name: str, sensor_type: str = "camera") -> None:
        # load sensor parameters from yaml file
        super().__init__(sensor_name, sensor_type)
        self.pan = 0
        self.tilt = 0
        self.xyz_offset = np.array([0, 0, 0])

        # Only dealing with depth data for now, TODO: add RGB data
        self.depth_width = self.params["depth"]["width"]
        self.depth_height = self.params["depth"]["height"]

        # Some optical sensors only provide diagonal field of view, get horizontal and vertical from diagonal
        try:
            self.vfov = self.params["depth"]["vfov"]
            self.hfov = self.params["depth"]["hfov"]
        except KeyError:
            self.dfov = self.params["depth"]["dfov"]
            self.hfov, self.vfov = ch.get_fov_from_dfov(self.depth_width, self.depth_height, self.dfov)
        self.near_val = self.params["depth"]["near_plane"]
        self.far_val = self.params["depth"]["far_plane"]

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
            fov=self.vfov, aspect=(self.depth_width / self.depth_height), nearVal=self.near_val, farVal=self.far_val
        )
        return


def main():
    import pprint as pp
    pbutils = PyBUtils(renders=False)
    camera = Camera(pbutils, sensor_name="realsense_d435i")
    pp.pprint(camera.depth_pixel_coords)
    pp.pprint(camera.depth_film_coords)
    pp.pprint(camera.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
