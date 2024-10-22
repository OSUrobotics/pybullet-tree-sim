#!/usr/bin/env python3
"""Base class for a ToF Camera. Inherits functionality from base Camera class"""

from pybullet_tree_sim import TOFS_PATH
from pybullet_tree_sim.camera import Camera
import pybullet_tree_sim.utils.yaml_utils as yutils


import os
from zenlog import log

class TimeOfFlight(Camera):
    def __init__(self, type: str) -> None:

        self.params = self._load_tof_params(type=type)
        print(self.params)
        return
        
        
    def _load_tof_params(self, type: str) -> dict:
        """
        @param type: The type of ToF to be used.
        """
        tof_conf_path = os.path.join(TOFS_PATH, f"{type}.yaml")
        if os.path.exists(tof_conf_path):
            log.info(f"Loading camera configuration from {tof_conf_path}")
            tof_conf_content = yutils.load_yaml(tof_conf_path)
            if tof_conf_content is not None:
                return tof_conf_content
            else:
                raise Exception(f"Failed to load camera configuration from {tof_conf_path}")
        else:
            raise FileNotFoundError(f"Camera configuration not found at {tof_conf_path}")
        
        
def main():
    
    tof = TimeOfFlight(type="vl53l8cx")
    
    return
    
    
if __name__ == "__main__":
    main()