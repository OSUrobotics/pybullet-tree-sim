#!/usr/bin/env python3
# Requires: pip install trimesh pyrender numpy
from pybullet_tree_sim import MESHES_PATH
from pybullet_tree_sim.utils.mesh_objects import MeshObjects
from pybullet_tree_sim.sensors.optical_sensor import OpticalSensor
import trimesh
import pyrender
import numpy as np
import os

import open3d as o3d


class RenderScene:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        """Initialize scene with a colored trimesh

        :param mesh: Mesh with face_colors or vertex_colors already set
        :type mesh: trimesh.Trimesh
        """
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])

        # Convert timesh to pyrender mesh
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.mesh_node: pyrender.Node = self.scene.add(self.mesh)

        # Active camera nodes
        self.sensor_nodes = {}

        # Renderer
        self.renderer = None
        return

    def add_camera(
        self,
        camera: OpticalSensor,
        pose: np.ndarray,
        camera_name: str,
        mode: str = "rgb",
    ) -> None:
        """Add a camera to the PyRender scene
        https://pyrender.readthedocs.io/en/latest/generated/pyrender.camera.IntrinsicsCamera.html#pyrender.camera.IntrinsicsCamera

        :param camera: An object derived from base class OpticalSensor. Options include DepthSensor, RGBCamera, DepthCamera
        :type camera: OpticalSensor
        :param pose: A matrix describing the camera pose to world, RT
        :type pose: np.ndarray
        :param camera_name: Name of the camera. Needed for adding/deleting cameras from the scene
        :type camera_name: str
        """
        # TODO: change this to accommodate depth, rgb, rgbd
        camera_intrinsics = camera.get_camera_intrinsics()[mode]

        pyr_camera = pyrender.IntrinsicsCamera(
            fx=camera_intrinsics["fx"],
            fy=camera_intrinsics["fy"],
            cx=camera_intrinsics["cx"],
            cy=camera_intrinsics["cy"],
            znear=camera_intrinsics["znear"],
            zfar=camera_intrinsics["zfar"],
            name=camera.sensor_name,
        )

        camera_node = self.scene.add(pyr_camera, pose=pose)
        self.sensor_nodes[camera_name] = camera_node

        return

    def remove_camera(self, camera_name: str) -> None:
        """Remove a camera from the PyRender scene"""
        if camera_name in self.sensor_nodes:
            self.scene.remove_node(self.sensor_nodes[camera_name])
            del self.sensor_nodes[camera_name]
        return

    def update_camera_pose(self, camera_name: str, pose: np.ndarray) -> None:
        """Update the pose of an existing camera"""
        if camera_name in self.sensor_nodes:
            self.scene.set_pose(self.sensor_nodes[camera_name], pose=pose)
        return

    def render_visual(self) -> None:
        """Render the scene using the interactive viewer"""
        pyrender.Viewer(self.scene)
        return

    def render_optical_sensor(self, sensor: OpticalSensor) -> None:
        if sensor.sensor_name not in self.sensor_nodes:
            raise ValueError(f"Camera '{sensor.sensor_name}' not found in scene.")

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=sensor.depth_width, viewport_height=sensor.depth_height
        )
        color, depth = self.renderer.render(
            scene=self.scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.SKIP_CULL_FACES
        )
        return color, depth

    def render_lidar_scan(self, mesh, lidar_sensor, extrinsics):
        """
        Cast rays for lidar simulation.

        Returns:
        --------
        points : np.ndarray (N, 3)
            Point cloud in world coordinates
        """
        azimuths, elevations = lidar_sensor.get_scan_pattern()

        # Use trimesh ray casting
        ray_origins = []
        ray_directions = []

        for az in azimuths:
            for el in elevations:
                # Convert spherical to Cartesian
                direction = np.array(
                    [
                        np.cos(el) * np.cos(az),
                        np.cos(el) * np.sin(az),
                        np.sin(el),
                    ]
                )
                ray_origins.append(extrinsics[:3, 3])  # Camera position
                ray_directions.append(direction)

        # Cast all rays at once
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions
        )

        return locations  # Nx3 point cloud

    def cleanup(self) -> None:
        """Clean up renderer resources"""
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None

        return


def main():
    __here__ = os.path.dirname(os.path.dirname(__file__))
    mesh_path = f"{__here__}/meshes/trees/LPy_envy_00000.ply"
    color_mesh = MeshObjects.color_unique_faces(mesh_path=mesh_path)
    MeshObjects.export_obj(mesh=color_mesh, filename="test.obj")
    scene = RenderScene(color_mesh)
    scene.render_visual()
    return


if __name__ == "__main__":
    main()


# 2) Duplicate vertices per-face so each face can get a flat color (no vertex-sharing)
#    This ensures a unique set of vertices per triangle so color doesn't interpolate across adjacent faces.
# verts_per_face = tm.vertices[faces]              # shape (n_faces, 3, 3)
# new_vertices = verts_per_face.reshape(-1, 3)     # (n_faces*3, 3)
# new_faces = np.arange(len(new_vertices)).reshape(-1, 3)  # (n_faces, 3)

# # 3) Encode face IDs into 24-bit RGB colors (reserve a background id 0 if you like)
# ids = np.arange(n_faces, dtype=np.uint32)
# r = (ids >> 16) & 0xFF
# g = (ids >> 8) & 0xFF
# b = ids & 0xFF
# face_colors = np.stack([r, g, b, np.full_like(r, 255)], axis=1).astype(np.uint8)  # RGBA per-face

# # Because we duplicated vertices (3 per face), create per-vertex colors by repeating each face color 3x
# vertex_colors = np.repeat(face_colors, 3, axis=0)   # shape (n_faces*3, 4)

# # 4) Make a new trimesh with per-vertex colors
# flat_tm = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
# flat_tm.visual.vertex_colors = vertex_colors

# # 5) Convert to pyrender mesh and render offscreen
# pyr_mesh = pyrender.Mesh.from_trimesh(flat_tm, smooth=False)
# scene = pyrender.Scene()
# scene.add(pyr_mesh)

# # Basic camera setup (adjust intrinsics to match your simulated depth camera)
# w, h = 640, 480
# camera_pose = np.identity(4)
# from scipy.spatial.transform import Rotation
# camera_pose[:3, :3] = Rotation.from_euler('xyz', [4*np.pi/3, 0,0]).as_matrix()
# camera_pose[:3, 3] = [0,-2, 0.5]
# print(camera_pose)
# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# cam_node = scene.add(camera, pose=camera_pose)

# # Add a light so the renderer produces colors clearly
# light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
# scene.add(light, pose=np.eye(4))

# pyrender.Viewer(scene, use_raymond_lighting=True)

# rdr = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
# color, depth = rdr.render(scene)

# # 6) Decode color to triangle IDs (background where color == [0,0,0] will decode to 0)
# color_uint8 = (color * 255.0).astype(np.uint8) if color.dtype == np.float32 else color.astype(np.uint8)
# ids_img = (color_uint8[:,:,0].astype(np.uint32) << 16) | (color_uint8[:,:,1].astype(np.uint32) << 8) | color_uint8[:,:,2].astype(np.uint32)

# seen_ids = np.unique(ids_img)
# # Remove background id if necessary (e.g., if background was 0)
# seen_ids = seen_ids[seen_ids < n_faces]   # safety filter if you used 0 for background
# print("Number of seen triangles:", len(seen_ids))
# print("Some seen triangle ids:", seen_ids[:20])
