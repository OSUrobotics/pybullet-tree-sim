#!/usr/bin/env python3
from __future__ import annotations

"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.utils import camera_helpers as ch
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.sensors.depth_sensor import DepthSensor
from pybullet_tree_sim.sensors.rgb_sensor import RGBSensor

import numpy as np
import os

from zenlog import log


class Camera(RGBSensor, DepthSensor):
    def __init__(self, sensor_type: str = "camera", *args, **kwargs) -> None:
        super().__init__(sensor_type=sensor_type, *args, **kwargs)
        # TODO: check if camera has depth, check here, not in depth class. How to avoid inheritance without depth? Bool flag passed to super?

        return


def main():
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    camera = Camera(pbclient=pbutils.pbclient, sensor_name="realsense_d435i")
    pp.pprint(camera.params)
    # pp.pprint(camera.depth_pixel_coords)
    # pp.pprint(camera.depth_film_coords)
    # pp.pprint(camera.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
