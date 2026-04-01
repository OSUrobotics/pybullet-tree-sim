#!/usr/bin/env python3
from __future__ import annotations

"""Base class for a ToF Camera. Inherits functionality from DepthSensor class"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.sensors.sensor_types import Modality
from pybullet_tree_sim.sensors.depth_sensor import DepthSensor
# from pybullet_tree_sim.utils import camera_helpers
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
# import pybullet_tree_sim.utils.yaml_utils as yutils
from pybullet_utils import bullet_client as bc

import numpy as np
import os

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TimeOfFlight(DepthSensor):
    def __init__(self, sensor_type: str = "tof", *args, **kwargs) -> None:
        """Builds a ToF camera object from a base Camera class"""
        super().__init__(sensor_type=sensor_type, *args, **kwargs)

        # TODO: move to base Sensor class?
        base_offset = self.params["depth"]["sensing_unit_offset"]
        self.base__sensing_unit_xyz_offset = (
            base_offset["x"],  # x
            base_offset["y"],  # y
            base_offset["z"],  # z
        )
        self.base__sensing_unit_rpy_offset = (
            base_offset["roll"],  # r
            base_offset["pitch"],  # p
            base_offset["yaw"],  # y
        )

        return

    def read(
        self, robot: Robot, pbclient: bc.BulletClient, mode: Modality = Modality.DEPTH
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read data from the ToF sensor."""
        return super().read(robot=robot, pbclient=pbclient, mode=mode)


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
