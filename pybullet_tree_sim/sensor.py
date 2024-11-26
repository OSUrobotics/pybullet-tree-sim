#!/usr/bin/env python3

from pybullet_tree_sim import CONFIG_PATH
import pybullet_tree_sim.utils.yaml_utils as yutils

import os
from zenlog import log


class Sensor:
    def __init__(self, sensor_name: str, sensor_type: str) -> None:
        """
        Initialize the sensor object.

        @param name: The model name of sensor to be used.
        @param sensor_type: The type of sensor to be used. Current options are: ['camera', 'tof']
        @return: None
        """
        sensor_type = sensor_type.strip().lower()
        self.sensor_path = os.path.join(CONFIG_PATH, "description", sensor_type)
        self.params = self._load_params(sensor_name=sensor_name, sensor_type=sensor_type)
        self.tf_frame: str
        self.tf_id: int
        return

    def _load_params(self, sensor_name: str, sensor_type) -> dict:
        """
        @param name: The model name of sensor to be used.
        @param sensor_type: The type of sensor to be used. Current options are: ['camera', 'tof']
        @return: A dictionary containing the sensor parameters.
        """
        sensor_config_path = os.path.join(self.sensor_path, f"{sensor_name}.yaml")

        if os.path.exists(sensor_config_path):
            log.info(f"Loading sensor configuration from {sensor_config_path}")
            config_content = yutils.load_yaml(sensor_config_path)
            if config_content is not None:
                return config_content
            else:
                raise Exception(f"Failed to load sensor configiguration from {sensor_config_path}")
        else:
            raise FileNotFoundError(f"Sensor configuration not found at {sensor_config_path}")


def main():

    sensor = Sensor(sensor_name="realsense_d435i", sensor_type="camera")
    print(sensor.params)

    sensor = Sensor(sensor_name="vl53l8cx", sensor_type="tof")
    print(sensor.params)

    sensor = Sensor(sensor_name="vl6180", sensor_type="tof")
    print(sensor.params)

    return


if __name__ == "__main__":
    main()
