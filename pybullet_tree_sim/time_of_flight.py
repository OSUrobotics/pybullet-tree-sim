#!/usr/bin/env python3
"""Base class for a ToF Camera. Inherits functionality from base Camera class"""

from pybullet_tree_sim import TOFS_PATH
from pybullet_tree_sim.camera import Camera
import pybullet_tree_sim.utils.yaml_utils as yutils


import os
from zenlog import log

class TimeOfFlight(Camera):
    def __init__(self, sensor_name: str, sensor_type: str = "tof") -> None:
        """Builds a ToF camera object from a base Camera class"""
        super().__init__(sensor_name=sensor_name, sensor_type=sensor_type)
        return




def main():

    tof = TimeOfFlight(sensor_name="vl53l8cx")
    print(tof.params)

    return


if __name__ == "__main__":
    main()
