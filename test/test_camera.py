#!/usr/bin/env python3
from pybullet_tree_sim.camera import Camera
from pybullet_tree_sim.utils.pyb_utils import PyBUtils

import numpy as np
import pprint as pp
import unittest


class TestCamera(unittest.TestCase):
    pbutils = PyBUtils(renders=False)
    camera = Camera(pbutils.pbclient, sensor_name="realsense_d435i")

    def test_aspect_ratio(self):
        """Test that aspect ratio is calculated correctly."""
        aspect_ratio = self.camera.depth_width / self.camera.depth_height
        self.assertAlmostEqual(aspect_ratio, self.camera.depth_width / self.camera.depth_height)
        return

    def test_camera_initialization(self):
        """Test that camera parameters are initialized correctly."""
        self.assertEqual(self.camera.depth_width, self.camera.params["depth"]["width"])
        self.assertEqual(self.camera.depth_height, self.camera.params["depth"]["height"])
        self.assertAlmostEqual(self.camera.vfov, self.camera.params["depth"]["vfov"])
        self.assertAlmostEqual(self.camera.hfov, self.camera.params["depth"]["hfov"])
        self.assertAlmostEqual(self.camera.near_val, self.camera.params["depth"]["near_plane"])
        self.assertAlmostEqual(self.camera.far_val, self.camera.params["depth"]["far_plane"])
        return
    
    def test_custom_resolution(self):
        """Test camera initialization with test_camera.yaml."""
        custom_camera = Camera(self.pbutils.pbclient, sensor_name="test_camera")
        custom_camera.depth_width = 640
        custom_camera.depth_height = 480
        custom_camera.depth_pixel_coords = np.array(
            list(np.ndindex((custom_camera.depth_width, custom_camera.depth_height))), dtype=int
        )
        self.assertEqual(custom_camera.depth_pixel_coords.shape, (640 * 480, 2))
        return

    def test_depth_pixel_coords_range(self):
        """Test that the pixel coordinates are in the range x=[0, depth_width] and y=[0, depth_height]."""
        self.assertTrue(self.camera.depth_pixel_coords[0, 0] == 0)
        self.assertTrue(self.camera.depth_pixel_coords[0, 1] == 0)
        self.assertTrue(self.camera.depth_pixel_coords[-1, 0] == self.camera.depth_width - 1)
        self.assertTrue(self.camera.depth_pixel_coords[-1, 1] == self.camera.depth_height - 1)
        return

    def test_depth_film_coords_range(self):
        """Test that the film coordinates are in the range [-1, 1]."""
        self.assertTrue(self.camera.depth_film_coords[0, 0] > -1)
        self.assertTrue(self.camera.depth_film_coords[0, 1] > -1)
        self.assertTrue(self.camera.depth_film_coords[-1, 0] < 1)
        self.assertTrue(self.camera.depth_film_coords[-1, 1] < 1)
        return
    
    def test_depth_film_coords_normalization(self):
        """Test that film coordinates are normalized to [-1, 1]."""
        coords = self.camera.depth_film_coords
        self.assertTrue(np.all(coords >= -1))
        self.assertTrue(np.all(coords <= 1))
        return

    def test_xy_depth_projection(self):
        """Test whether a depth pixel has be adequately scaled to xy"""
        # depth_width = 8
        # depth_height = 8
        # xy_pixels_order_C = np.array(list(np.ndindex((self.camera.depth_width, self.camera.depth_height))), dtype=int)
        # print(xy_pixels_order_C)
        print(self.camera.depth_pixel_coords)

        return


if __name__ == "__main__":
    unittest.main()
