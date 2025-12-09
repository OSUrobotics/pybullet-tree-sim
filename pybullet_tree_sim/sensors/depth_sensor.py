#!/usr/bin/env python3
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
import pybullet_tree_sim.utils.camera_helpers as ch
import numpy as np

from typing import Union


class DepthSensor(OpticalSensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pbclient = kwargs.get("pbclient")

        depth_params = self.params["depth"]

        # Get depth sensor parameters
        self.depth_width = depth_params["width"]
        self.depth_height = depth_params["height"]
        # Some optical sensors only provide diagonal field of view, get horizontal and vertical from diagonal
        try:
            self.depth_vfov = depth_params["vfov"]
            self.depth_hfov = depth_params["hfov"]
        except KeyError:
            self.depth_dfov = depth_params["dfov"]
            self.depth_hfov, self.depth_vfov = ch.get_fov_from_dfov(
                self.depth_width, self.depth_height, self.depth_dfov
            )
        self.near_val = depth_params["near_plane"]
        self.far_val = depth_params["far_plane"]

        # Pixel coordinates, indexed by depth_width, depth_height, nx1 array COLUMN MAJOR
        self.depth_pixel_coords = np.array(list(np.ndindex((self.depth_width, self.depth_height))), dtype=int)
        # Film coordinates projected to [-1, 1], nx1 array COLUMN MAJOR
        self.depth_film_coords = (
            2
            * (self.depth_pixel_coords + np.array([0.5, 0.5]) - np.array([self.depth_width / 2, self.depth_height / 2]))
            / np.array([self.depth_width, self.depth_height])
        )
        # Depth projection matrix from camera intrinsics
        self.depth_proj_mat = pbclient.computeProjectionMatrixFOV(
            fov=self.depth_vfov,
            aspect=(self.depth_width / self.depth_height),
            nearVal=self.near_val,
            farVal=self.far_val,
        )
        return

    def get_camera_intrinsics(self) -> dict:  # TODO: change to *args ?
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
                znear=self.near_val,
                zfar=self.far_val,
            )
        )


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
