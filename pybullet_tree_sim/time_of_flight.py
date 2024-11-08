#!/usr/bin/env python3
"""Base class for a ToF Camera. Inherits functionality from base Camera class"""

from pybullet_tree_sim.camera import Camera
from pybullet_tree_sim.utils import camera_helpers
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.yaml_utils as yutils

import os
from zenlog import log


class TimeOfFlight(Camera):
    def __init__(self, pbutils: PyBUtils, sensor_name: str, sensor_type: str = "tof") -> None:
        """Builds a ToF camera object from a base Camera class"""
        super().__init__(pbutils=pbutils, sensor_name=sensor_name, sensor_type=sensor_type)

        return


def main():
    pbutils = PyBUtils(renders=False)
    tof = TimeOfFlight(pbutils, sensor_name="vl53l8cx")
    print(tof.depth_proj_mat)

    return


if __name__ == "__main__":
    main()
