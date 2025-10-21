#!/usr/bin/env python3
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
import pybullet_tree_sim.utils.camera_helpers as ch

import numpy as np


class RGBCamera(OpticalSensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        pbclient = kwargs.get("pbclient")

        # Get rgb sensor parameters
        self.rgb_width = self.params["rgb"]["width"]
        self.rgb_height = self.params["rgb"]["height"]

        # Some optical sensors only provide diagonal field of view, get horizontal and vertical from diagonal
        try:
            self.rgb_vfov = self.params["rgb"]["vfov"]
            self.rgb_hfov = self.params["rgb"]["hfov"]
        except KeyError:
            self.rgb_dfov = self.params["rgb"]["dfov"]
            self.rgb_hfov, self.rgb_vfov = ch.get_fov_from_dfov(
                self.rgb_width, self.rgb_height, self.rgb_dfov
            )

        self.near_val = self.params["rgb"]["near_plane"]
        self.far_val = self.params["rgb"]["far_plane"]

        # Pixel coordinates, indexed by depth_width, depth_height, nx1 array COLUMN MAJOR
        self.rgb_pixel_coords = np.array(
            list(np.ndindex((self.rgb_width, self.rgb_height))), dtype=int
        )
        # Film coordinates projected to [-1, 1], nx1 array COLUMN MAJOR
        self.rgb_film_coords = (
            2
            * (
                self.rgb_pixel_coords
                + np.array([0.5, 0.5])
                - np.array([self.rgb_width / 2, self.rgb_height / 2])
            )
            / np.array([self.rgb_width, self.rgb_height])
        )

        return

    def get_camera_intrinsics(self) -> dict:  # TODO: change to *args ?
        """Convert depth sensor parameters to standard camera intrinsics.
        :return: Dict with camera intrinsics. Keys -- fx, fy, cx, cy, width, height, znear, zfar
        :rtype: dict
        """
        # Calculate focal lengths from FOV
        fx = self.rgb_width / (2 * np.tan(np.radians(self.rgb_hfov) / 2))
        fy = self.rgb_height / (2 * np.tan(np.radians(self.rgb_vfov) / 2))

        # Principal point at image center
        cx = self.rgb_width / 2
        cy = self.rgb_height / 2

        return dict(
            rgb=dict(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=self.rgb_width,
                height=self.rgb_height,
                znear=self.near_val,
                zfar=self.far_val,
            )
        )


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    sensor = RGBCamera(
        pbclient=pbutils.pbclient,
        sensor_name="realsense_d435i",
        sensor_type="camera",
    )
    pp.pprint(sensor.params)
    # pp.pprint(sensor.depth_film_coords)
    # pp.pprint(sensor.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
