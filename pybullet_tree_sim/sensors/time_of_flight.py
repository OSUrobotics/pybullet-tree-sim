#!/usr/bin/env python3
"""Base class for a ToF Camera. Inherits functionality from DepthSensor class"""
from pybullet_tree_sim.sensors.depth_sensor import DepthSensor
from pybullet_tree_sim.utils import camera_helpers
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
import pybullet_tree_sim.utils.yaml_utils as yutils

import os
from zenlog import log


class TimeOfFlight(DepthSensor):
    def __init__(self, sensor_type: str = "tof", *args, **kwargs) -> None:
        """Builds a ToF camera object from a base Camera class"""
        super().__init__(sensor_type=sensor_type, *args, **kwargs)

        return


def main():
    import pprint as pp
    pbutils = PyBUtils(renders=False)
    tof = TimeOfFlight(pbclient=pbutils.pbclient, sensor_name="vl53l8cx")
    # print(tof.depth_proj_mat)
    pp.pprint(tof.params)
    return


if __name__ == "__main__":
    main()
    ## startswith
    # ti.timeit("from pybullet_tree_sim.time_of_flight import TimeOfFlight; from pybullet_tree_sim.utils.pyb_utils import PyBUtils; pbutils=PyBUtils(renders=False); tofs = {'tof0': TimeOfFlight(pbutils.pbclient, sensor_name='vl53l8cx')}; list(tofs.keys())[0].startswith('tof')", number=1)
    ## isinstance
    # ti.timeit("from pybullet_tree_sim.time_of_flight import TimeOfFlight; from pybullet_tree_sim.utils.pyb_utils import PyBUtils; pbutils=PyBUtils(renders=False); tofs = {'tof0': TimeOfFlight(pbutils.pbclient, sensor_name='vl53l8cx')}; isinstance(tofs['tof0'], TimeOfFlight)", number=1)
    # 
