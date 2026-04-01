#!/usr/bin/env python3
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot

from abc import ABC, abstractmethod
from pybullet_tree_sim import CONFIG_PATH
from pybullet_tree_sim.sensors import sensor_types as st
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils import yaml_utils as yutils
from pybullet_utils import bullet_client as bc


import numpy as np
import os

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Sensor(ABC):
    def __init__(
        self,
        sensor_type: str,
        sensor_model: str,
        sensor_name: str,
        tf_frame: str = None,
        tf_id: int = None,
        *args,
        **kwargs,
    ) -> None:
        """Abstract sensor base class

        :param sensor_type: The type of sensor to be used. Current options are: ['camera', 'tof']
        :type sensor_type: str
        :param sensor_model: The model name of sensor to be used.
        :type sensor_model: str
        :param sensor_name: The local name of sensor to be used. This name is reflected in the URDF tree.
        :type sensor_name: str
        :param tf_frame: TF frame name in the URDF tree, defaults to None
        :type tf_frame: str, optional
        :param tf_id: PyBullet TF ID, defaults to None
        :type tf_id: int, optional
        """
        self.sensor_type: str = sensor_type.strip().lower()
        self.model: str = sensor_model.strip().lower()
        self.sensor_name: str = sensor_name.strip().lower()

        # Get sensor YAML params
        self.sensor_path: str = os.path.join(CONFIG_PATH, "sensors", self.sensor_type)
        self.params: dict = self._load_params()

        # Intrinsic attributes
        self.data_rate_hz: float = self.params["data_rate_hz"]
        self.data_type: st.DataType = st.parse_data_type(self.params["data_type"])
        self.modalities: set = st.data_type_modalities(self.data_type)

        # Extrinsic attributes
        self.tf_frame: str = tf_frame
        self.tf_id: int = tf_id
        self.xyz_offset: np.ndarray = np.zeros(3, dtype=float)
        self.rpy_offset: np.ndarray = np.zeros(3, dtype=float)

        # Physical attributes
        self.mass: float = self.params.get("mass", None)
        dimensions: dict = self.params.get("dimensions", {})
        self.shape: str = dimensions.get("shape", None)
        self.dimensions: tuple[float] = (
            dimensions.get("x", None),
            dimensions.get("y", None),
            dimensions.get("z", None),
        )
        return

    def _load_params(self) -> dict:
        """Loads parameters from the corresponding sensor YAML file.

        :raises Exception: _description_
        :raises FileNotFoundError: If raised, config file doesn't exist.
        :return: A parameter dictionary describing the sensor.
        :rtype: dict
        """
        sensor_config_path = os.path.join(self.sensor_path, f"{self.model}.yaml")

        if os.path.exists(sensor_config_path):
            logger.info(f"Loading sensor configuration from {sensor_config_path}")
            config_content = yutils.load_yaml(sensor_config_path)
            if config_content is not None:
                return config_content
            else:
                raise Exception(f"Failed to load sensor configiguration from {sensor_config_path}")
        else:
            raise FileNotFoundError(f"Sensor configuration not found at {sensor_config_path}")

    @abstractmethod
    def read(self, robot: Robot, pbclient: bc.BulletClient, **kwargs) -> dict:
        """Abstract method to read data from the sensor.

        :raises NotImplementedError: If raised, the method is not implemented in the subclass.
        """
        raise NotImplementedError("The read method must be implemented by the subclass.")


def main():
    import pprint as pp

    sensor = Sensor(sensor_model="realsense_d435i", sensor_type="depth_camera")
    pp.pprint(sensor.params)

    sensor = Sensor(sensor_model="vl53l8cx", sensor_type="tof")
    pp.pprint(sensor.params)

    sensor = Sensor(sensor_model="vl6180", sensor_type="tof")
    pp.pprint(sensor.params)

    return


if __name__ == "__main__":
    main()
