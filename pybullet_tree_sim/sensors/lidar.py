#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pybullet_tree_sim.robot import Robot

import logging

import numpy as np
from pybullet_utils import bullet_client as bc
from scipy.spatial.transform import Rotation as R

import pybullet_tree_sim.utils.logging_conf  # noqa: F401
from pybullet_tree_sim.sensors.sensor import Sensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Lidar(Sensor):
    """LIDAR sensor simulation using PyBullet raycasting.

    This sensor generates a point cloud by casting rays from the LIDAR position
    in all directions within its field of view and detecting intersections with
    objects in the scene.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize LIDAR sensor.

        Extracts LIDAR-specific parameters from the sensor YAML configuration
        file and sets up ray generation parameters.
        """
        super().__init__(*args, **kwargs)

        # Extract LIDAR-specific parameters
        lidar_params = self.params.get("lidar", self.params)

        # FOV parameters (in degrees)
        self.theta = lidar_params.get("theta", 360)  # Horizontal FOV
        self.phi = lidar_params.get("phi", 59)  # Vertical FOV

        # Resolution (number of rays)
        self.width = lidar_params.get("width", 5760)  # Horizontal resolution
        self.height = lidar_params.get("height", 2880)  # Vertical resolution

        # Distance parameters (in meters)
        self.near_plane = lidar_params.get("near_plane", 0.02)
        self.far_plane = lidar_params.get("far_plane", 40)

        # Wavelength (nm) - useful for simulation fidelity
        self.wavelength = lidar_params.get("wavelength", 905)

        logger.debug(f"LIDAR '{self.sensor_name}' initialized:")
        logger.debug(f"  FOV (theta x phi): {self.theta}° x {self.phi}°")
        logger.debug(f"  Resolution (width x height): {self.width} x {self.height}")
        logger.debug(f"  Range: {self.near_plane}m to {self.far_plane}m")
        logger.debug(f"  Wavelength: {self.wavelength}nm")
        logger.debug(f"  Total rays: {self.width * self.height}")

        return

    def _generate_ray_directions(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate ray origins and directions for the LIDAR.

        Creates a grid of rays distributed across the LIDAR's field of view.
        Theta (horizontal) and Phi (vertical) angles are converted to 3D
        direction vectors in the sensor's local coordinate frame.

        Returns:
            tuple: (ray_origins, ray_directions)
                - ray_origins: (N, 3) array of ray starting positions (zeros)
                - ray_directions: (N, 3) array of normalized ray direction vectors
        """
        # Define angle ranges
        theta_start = -self.theta / 2
        theta_end = self.theta / 2
        phi_start = -self.phi / 2
        phi_end = self.phi / 2

        # Generate angle grids
        theta_angles = np.linspace(theta_start, theta_end, self.width, endpoint=True)
        phi_angles = np.linspace(phi_start, phi_end, self.height, endpoint=True)

        # Create meshgrid of angles
        theta_grid, phi_grid = np.meshgrid(theta_angles, phi_angles, indexing="ij")

        # Flatten the grids
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()

        # Convert to radians
        theta_rad = np.radians(theta_flat)
        phi_rad = np.radians(phi_flat)

        # Generate direction vectors in the LIDAR frame
        # LIDAR looks forward (+Z) with:
        # - Theta (horizontal): rotation around Z axis
        # - Phi (vertical): rotation around Y axis
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(phi_rad)
        z = np.cos(theta_rad) * np.cos(phi_rad)

        ray_directions = np.stack([x, y, z], axis=1)

        # Normalize directions
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / (norms + 1e-10)

        # Ray origins initialized to zero (will be set to sensor position)
        ray_origins = np.zeros((len(theta_flat), 3), dtype=np.float32)

        return ray_origins, ray_directions

    def _create_sensor_transform(self, world_position: np.ndarray, world_orientation: tuple) -> np.ndarray:
        """Create a 4x4 transformation matrix for the sensor.

        Converts the sensor's position and orientation in the world frame
        along with any frame offsets into a single homogeneous transformation
        matrix.

        Args:
            world_position: (3,) array of sensor position in world frame [x, y, z]
            world_orientation: Quaternion (x, y, z, w) of sensor orientation

        Returns:
            np.ndarray: 4x4 transformation matrix (world_T_sensor)
        """
        # Create base transformation matrix
        tf = np.identity(4, dtype=np.float32)

        # Convert quaternion to rotation matrix
        rot = R.from_quat([world_orientation[0], world_orientation[1], world_orientation[2], world_orientation[3]])
        tf[:3, :3] = rot.as_matrix()

        # Set translation part
        tf[:3, 3] = world_position

        # Apply any frame offset (from URDF to sensor frame)
        offset_tf = np.identity(4, dtype=np.float32)
        offset_tf[:3, 3] = self.xyz_offset

        # Apply RPY offset rotation if present
        if np.any(self.rpy_offset != 0):
            offset_rot = R.from_euler("xyz", self.rpy_offset)
            offset_tf[:3, :3] = offset_rot.as_matrix()

        # Compose transformations
        tf = tf @ offset_tf
        return tf

    def read(self, robot: Robot, pbclient: bc.BulletClient, **kwargs) -> dict:
        """Read data from the LIDAR sensor using raycasting.

        Performs the following steps:
        1. Get the LIDAR's current position and orientation from the robot
        2. Generate rays in the sensor's local coordinate frame
        3. Transform rays to the world frame
        4. Cast rays into the scene using PyBullet
        5. Filter results by distance range
        6. Return point cloud and metadata

        Args:
            robot: Robot object with pose information
            pbclient: PyBullet client for raycasting
            **kwargs: Additional arguments (unused)

        Returns:
            dict: Sensor reading containing:
                - 'points': (N, 3) point cloud in world coordinates
                - 'distances': (N,) distances from LIDAR origin
                - 'extrinsic_matrix': (4, 4) transformation matrix
                - 'ray_origins': (M, 3) ray origins in world frame
                - 'ray_directions': (M, 3) ray directions in world frame
                - 'num_rays_total': Total number of rays cast
                - 'num_rays_hit': Number of rays that hit objects
                - 'intrinsics': Dict with sensor parameters
        """
        # Get sensor pose from robot
        sensor_pos, sensor_orn = robot.get_current_pose(self.tf_id)
        sensor_pos = np.array(sensor_pos, dtype=np.float32)

        logger.debug(f"LIDAR '{self.sensor_name}' reading at pos={sensor_pos}")

        # Generate ray directions in sensor frame
        ray_origins_local, ray_directions_local = self._generate_ray_directions()

        # Create transformation matrix from sensor to world
        sensor_tf = self._create_sensor_transform(sensor_pos, sensor_orn)

        # Transform ray origins to world frame (all at sensor position)
        ray_origins = np.full_like(ray_origins_local, sensor_pos, dtype=np.float32)

        # Transform ray directions to world frame
        rotation_matrix = sensor_tf[:3, :3]
        ray_directions = ray_directions_local @ rotation_matrix.T

        # Normalize directions after transformation
        norms = np.linalg.norm(ray_directions, axis=1, keepdims=True)
        ray_directions = ray_directions / (norms + 1e-10)

        # Calculate ray endpoints
        ray_ends = ray_origins + ray_directions * self.far_plane

        logger.debug(f"Casting {len(ray_origins)} rays with {self.width}x{self.height} resolution")

        # Perform batch raycasting
        results = pbclient.rayTestBatch(ray_origins, ray_ends, numThreads=4)

        # Process results and extract point cloud
        points = []
        distances = []

        for i, result in enumerate(results):
            obj_id, link_id, fraction, hit_pos, hit_normal = result

            # Check if ray hit something (obj_id != -1 means valid hit)
            if obj_id != -1:
                hit_pos = np.array(hit_pos, dtype=np.float32)

                # Calculate distance from ray origin to hit point
                distance = np.linalg.norm(hit_pos - ray_origins[i])

                # Filter by near and far planes
                if self.near_plane <= distance <= self.far_plane:
                    points.append(hit_pos)
                    distances.append(distance)

        # Convert to numpy arrays
        if len(points) > 0:
            points = np.array(points, dtype=np.float32)
            distances = np.array(distances, dtype=np.float32)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            distances = np.empty(0, dtype=np.float32)

        logger.debug(f"LIDAR '{self.sensor_name}' captured {len(points)}/{len(ray_origins)} points")

        # Create results dictionary
        results_dict = {
            "points": points,
            "distances": distances,
            "extrinsic_matrix": sensor_tf,
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "num_rays_total": len(ray_origins),
            "num_rays_hit": len(points),
            "intrinsics": {
                "theta": self.theta,
                "phi": self.phi,
                "width": self.width,
                "height": self.height,
                "near_plane": self.near_plane,
                "far_plane": self.far_plane,
                "wavelength": self.wavelength,
            },
        }

        return results_dict


def main():
    """Test the LIDAR implementation."""
    import pprint as pp

    from pybullet_tree_sim.utils.pyb_utils import PyBUtils

    pbutils = PyBUtils(renders=False)
    lidar = Lidar(pbclient=pbutils.pbclient, sensor_model="fjd_trion_p1", sensor_name="test_lidar", sensor_type="lidar")
    print("LIDAR initialized successfully")
    pp.pprint(lidar.params)
    print("\nLIDAR specifications:")
    print(f"  Horizontal FOV (theta): {lidar.theta}°")
    print(f"  Vertical FOV (phi): {lidar.phi}°")
    print(f"  Resolution: {lidar.width} x {lidar.height}")
    print(f"  Range: {lidar.near_plane}m to {lidar.far_plane}m")
    print(f"  Total rays: {lidar.width * lidar.height}")

    return


if __name__ == "__main__":
    main()
