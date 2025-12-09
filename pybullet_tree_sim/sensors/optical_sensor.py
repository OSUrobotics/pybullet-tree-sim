from abc import ABC, abstractmethod
from typing import Union

from pybullet_tree_sim.sensors.sensor import Sensor


class OpticalSensor(Sensor, ABC):
    """Base class for sensors with camera intrinsics"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pan = 0.0
        self.tilt = 0.0  # TODO: these are fillers, original settings are in YAML. Remove and just do URDF or keep?
        return

    @abstractmethod
    def get_camera_intrinsics(self):
        """Must be implmented by optical sensor subclasses"""
        pass


def main():
    opt_sensor = OpticalSensor()
    return


if __name__ == "__main__":
    main()
