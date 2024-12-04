#!/usr/bin/env python3
from pybullet_tree_sim.sensors.sensor import Sensor
import pybullet_tree_sim.utils.camera_helpers as ch
import numpy as np

class DepthSensor(Sensor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        pbclient = kwargs.get("pbclient")
                
        # # Only dealing with depth data for now, TODO: add RGB data
        self.depth_width = self.params["depth"]["width"]
        self.depth_height = self.params["depth"]["height"]

        # Some optical sensors only provide diagonal field of view, get horizontal and vertical from diagonal
        try:
            self.depth_vfov = self.params["depth"]["vfov"]
            self.depth_hfov = self.params["depth"]["hfov"]
        except KeyError:
            self.depth_dfov = self.params["depth"]["dfov"]
            self.depth_hfov, self.depth_vfov = ch.get_fov_from_dfov(self.depth_width, self.depth_height, self.depth_dfov)
        self.near_val = self.params["depth"]["near_plane"]
        self.far_val = self.params["depth"]["far_plane"]

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
            fov=self.depth_vfov, aspect=(self.depth_width / self.depth_height), nearVal=self.near_val, farVal=self.far_val
        )
        return
        
        
def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    sensor = DepthSensor(pbclient=pbutils.pbclient, sensor_name="vl53l8cx", sensor_type = "tof")
    pp.pprint(sensor.params)
    # pp.pprint(sensor.depth_pixel_coords)
    # pp.pprint(sensor.depth_film_coords)
    # pp.pprint(sensor.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
        
