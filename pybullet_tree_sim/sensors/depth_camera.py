#!/usr/bin/env python3
from __future__ import annotations

"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.utils import camera_helpers as ch
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
from pybullet_tree_sim.sensors.depth_sensor import DepthSensor
from pybullet_tree_sim.sensors.rgb_camera import RGBCamera

import numpy as np
import os

from zenlog import log


class DepthCamera(RGBCamera, DepthSensor):
    def __init__(
        self, sensor_type: str = "depth_camera", *args, **kwargs
    ) -> None:
        super().__init__(sensor_type=sensor_type, *args, **kwargs)

        rgb_params = self.params.get("rgb", {})
        depth_params = self.params.get("depth", {})

        # Initialize RGB attributes
        self.rgb_width = rgb_params["width"]
        self.rgb_height = rgb_params["height"]
        # ... RGB setup

        # Initialize depth attributes
        self.depth_width = depth_params["width"]
        self.depth_height = depth_params["height"]
        # ... depth setup

        return

    def get_camera_intrinsics(self, mode: str):
        """Get camera intrinsics
        @param mode: The sensor mode option. Choices are 'rgb' or 'depth'
        """
        if mode == "depth":
            fx = self.depth_width / (
                2 * np.tan(np.radians(self.depth_hfov) / 2)
            )
            fy = self.depth_height / (
                2 * np.tan(np.radians(self.depth_vfov) / 2)
            )
        return


def main():
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    dcamera = DepthCamera(
        pbclient=pbutils.pbclient, sensor_name="realsense_d435i"
    )
    print(DepthCamera.__mro__)
    print(dcamera.get_camera_intrinsics(mode="depth"))
    pp.pprint(dcamera.params)
    pp.pprint(dcamera.depth_pixel_coords)
    pp.pprint(dcamera.depth_film_coords)
    pp.pprint(dcamera.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
