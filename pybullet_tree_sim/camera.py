#!/usr/bin/env python3
"""Provides basic functions for a simulated camera.
Resources:
"""

from pybullet_tree_sim import CONFIG_PATH
import pybullet_tree_sim.utils.yaml_utils as yutils

import os

class Camera:
    def __init__(self, type: str) -> None:
        """
        Initialize the camera object.

        @param type: The type of camera to be used.
        """
        type = type.strip().lower()
        self.intinsics = self._load_camera_intrinsics(type=type)
        return
        
    def _load_camera_intrinsics(self, type: str) -> None:
        camera_config_file = os.path.join(CONFIG_PATH, f"{type}.yaml")
        camera_config_content = yutils.load_yaml(camera_config_file)
        return


def main():

    return


if __name__ == "__main__":
    camera = Camera(type="vl53l8cx")
    main()
