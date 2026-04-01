#!/usr/bin/env python3
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
import pybullet_tree_sim.utils.camera_helpers as ch
from pybullet_utils import bullet_client as bc

import numpy as np

import logging
import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DepthSensor(OpticalSensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pbclient = kwargs.get("pbclient")

        depth_params = self.params["depth"]
        self.depth_width = depth_params["width"]
        self.depth_height = depth_params["height"]
        try:
            self.depth_vfov = depth_params["vfov"]
            self.depth_hfov = depth_params["hfov"]
        except KeyError:
            self.depth_dfov = depth_params["dfov"]
            self.depth_hfov, self.depth_vfov = ch.get_fov_from_dfov(self.depth_width, self.depth_height, self.depth_dfov)
        self.z_near = depth_params["z_near"]
        self.z_far = depth_params["z_far"]

        depth_params = self.params["depth"]
        self.depth_pixel_coords = ch.get_pixel_coords(
            width=depth_params["width"],
            height=depth_params["height"],
        )
        self.depth_film_coords = ch.get_film_coords(
            width=depth_params["width"],
            height=depth_params["height"],
        )
        # self.depth_proj_mat = ch.get_projection_matrix(
        #     width=depth_params["width"],
        #     height=depth_params["height"],
        #     z_near=depth_params["z_near"],
        #     z_far=depth_params["z_far"],
        #     hfov=depth_params.get("hfov", None),
        #     vfov=depth_params.get("vfov", None),
        #     dfov=depth_params.get("dfov", None),
        # )
        
        # Depth projection matrix from camera intrinsics
        self.depth_proj_mat = pbclient.computeProjectionMatrixFOV(
            fov=self.depth_vfov,
            aspect=(self.depth_width / self.depth_height),
            nearVal=self.z_near,
            farVal=self.z_far,
        )
        return

    def get_optical_intrinsics(self) -> dict:  # TODO: change to *args ?
        """Convert depth sensor parameters to standard camera intrinsics.
        :return: Dict with camera intrinsics. Keys -- fx, fy, cx, cy, width, height, znear, zfar
        :rtype: dict
        """
        # Calculate focal lengths from FOV
        fx = self.depth_width / (2 * np.tan(np.radians(self.depth_hfov) / 2))
        fy = self.depth_height / (2 * np.tan(np.radians(self.depth_vfov) / 2))

        # Principal point at image center
        cx = self.depth_width / 2
        cy = self.depth_height / 2

        return dict(
            depth=dict(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=self.depth_width,
                height=self.depth_height,
                znear=self.z_near,
                zfar=self.z_far,
            )
        )

    # def read(self, robot: Robot, pbclient: bc.BulletClient):
    #     """Read data from the depth sensor."""
    #     # Implementation for reading data from the depth sensor
    #     pass


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    sensor = DepthSensor(pbclient=pbutils.pbclient, sensor_model="vl53l8cx", sensor_type="tof")
    print("Params:\n", sensor.params)
    pp.pprint(sensor.params)
    print("Pixel coords:\n")
    pp.pprint(sensor.depth_pixel_coords)
    print("Film coords:\n")
    pp.pprint(sensor.depth_film_coords)
    print("Proj matrix:\n")
    pp.pprint(sensor.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
