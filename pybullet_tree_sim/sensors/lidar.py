#!/usr/bin/env pythone
from pybullet_tree_sim.sensors.sensor import Sensor
from zenlog import log


class Lidar(Sensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pbclient = kwargs.get('pbclient')

        lidar_params = self.params.get("lidar", self.params)

        return
