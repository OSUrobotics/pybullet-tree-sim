from abc import ABC, abstractmethod
from typing import Union

from pybullet_tree_sim.sensors.sensor import Sensor


class OpticalSensor(Sensor, ABC):
    """Base class for sensors with camera intrinsics"""
    @abstractmethod
    def get_camera_intrinsics(self):
        """Must be implmented by optical sensor subclasses"""
        pass


def main():
    opt_sensor = OpticalSensor()
    return


if __name__ == "__main__":
    main()
