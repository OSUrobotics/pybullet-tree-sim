#!/usr/bin/env python3
from __future__ import annotations
from pybullet_tree_sim.sensors.depth_camera import DepthCamera
from pybullet_tree_sim.sensors.lidar import Lidar
from pybullet_tree_sim.sensors.rgb_camera import RGBCamera
from pybullet_tree_sim.sensors.sensor import Sensor
from pybullet_tree_sim.sensors.time_of_flight import TimeOfFlight

SENSOR_TYPE_MAP: dict[str, Sensor] = {
    "depth_camera": DepthCamera,
    "lidar": Lidar,
    "rgb_camera": RGBCamera,
    "tof": TimeOfFlight,
}
