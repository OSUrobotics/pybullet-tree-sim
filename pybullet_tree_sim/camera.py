#!/usr/bin/env python3
"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CAMERAS_PATH
from pybullet_tree_sim.sensor import Sensor
import os

from zenlog import log

class Camera(Sensor):
    def __init__(self, sensor_name: str, sensor_type: str='camera') -> None:
        """Builds a camera object from a base Sensor class"""
        super().__init__(sensor_name=sensor_name, sensor_type=sensor_type)
        return   


def main():
    camera = Camera(sensor_name="realsense_d435i")
    print(camera.params)
    return


if __name__ == "__main__":
    main()
