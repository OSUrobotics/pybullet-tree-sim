#!/usr/bin/env python3
"""Base class for a ToF Camera. Inherits functionality from base Camera class"""

from pybullet_tree_sim.camera import Camera
from pybullet_tree_sim.utils import camera_helpers
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.yaml_utils as yutils

import os
from zenlog import log


class TimeOfFlight(Camera):
    def __init__(self, pbclient, sensor_name: str, sensor_type: str = "tof") -> None:
        """Builds a ToF camera object from a base Camera class"""
        super().__init__(pbclient=pbclient, sensor_name=sensor_name, sensor_type=sensor_type)

        return


def main():
    pbutils = PyBUtils(renders=False)
    tof = TimeOfFlight(pbutils.pbclient, sensor_name="vl53l8cx")
    print(tof.depth_proj_mat)

    return


if __name__ == "__main__":
    main()
    ## startswith
    # ti.timeit("from pybullet_tree_sim.time_of_flight import TimeOfFlight; from pybullet_tree_sim.utils.pyb_utils import PyBUtils; pbutils=PyBUtils(renders=False); tofs = {'tof0': TimeOfFlight(pbutils.pbclient, sensor_name='vl53l8cx')}; list(tofs.keys())[0].startswith('tof')", number=1)
    ## isinstance
    # ti.timeit("from pybullet_tree_sim.time_of_flight import TimeOfFlight; from pybullet_tree_sim.utils.pyb_utils import PyBUtils; pbutils=PyBUtils(renders=False); tofs = {'tof0': TimeOfFlight(pbutils.pbclient, sensor_name='vl53l8cx')}; isinstance(tofs['tof0'], TimeOfFlight)", number=1)
    # 
