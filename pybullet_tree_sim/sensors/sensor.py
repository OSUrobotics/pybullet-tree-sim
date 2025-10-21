#!/usr/bin/env python3
from abc import ABC
from pybullet_tree_sim import CONFIG_PATH
import pybullet_tree_sim.utils.yaml_utils as yutils

import numpy as np
import os
from zenlog import log


class Sensor(ABC):
    def __init__(
        self, sensor_name: str, sensor_type: str, *args, **kwargs
    ) -> None:
        """
        Abstract sensor base class

        @param name: The model name of sensor to be used.
        @param sensor_type: The type of sensor to be used. Current options are: ['camera', 'tof']
        @return: None
        """
        # super().__init__(*args, **kwargs)
        sensor_name: str = sensor_name.strip().lower()
        sensor_type: str = sensor_type.strip().lower()
        self.sensor_path: str = os.path.join(
            CONFIG_PATH, "sensors", sensor_type
        )
        self.params: dict = self._load_params(
            sensor_name=sensor_name, sensor_type=sensor_type
        )
        self.tf_frame: str = None
        self.tf_id: int = None
        self.xyz_offset: np.ndarray = np.zeros(3, dtype=float)
        return

    def _load_params(self, sensor_name: str, sensor_type: str) -> dict:
        """
        @param name: The model name of sensor to be used.
        @param sensor_type: The type of sensor to be used. Current options are: ['camera', 'depth_camera', 'lidar', 'tof']
        @return: A dictionary containing the sensor parameters.
        """
        sensor_config_path = os.path.join(
            self.sensor_path, f"{sensor_name}.yaml"
        )

        if os.path.exists(sensor_config_path):
            log.info(f"Loading sensor configuration from {sensor_config_path}")
            config_content = yutils.load_yaml(sensor_config_path)
            if config_content is not None:
                return config_content
            else:
                raise Exception(
                    f"Failed to load sensor configiguration from {sensor_config_path}"
                )
        else:
            raise FileNotFoundError(
                f"Sensor configuration not found at {sensor_config_path}"
            )


def main():
    import pprint as pp

    sensor = Sensor(sensor_name="realsense_d435i", sensor_type="depth_camera")
    pp.pprint(sensor.params)

    sensor = Sensor(sensor_name="vl53l8cx", sensor_type="tof")
    pp.pprint(sensor.params)

    sensor = Sensor(sensor_name="vl6180", sensor_type="tof")
    pp.pprint(sensor.params)

    return


if __name__ == "__main__":
    main()
