#!/usr/bin/env pythone
from __future__ import annotations
from typing import TYPE_CHECKING
from pybullet_tree_sim.sensors.sensor import Sensor

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Lidar(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pbclient = kwargs.get('pbclient')

        lidar_params = self.params.get("lidar", self.params)

        return
