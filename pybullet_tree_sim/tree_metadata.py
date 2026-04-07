#!/usr/bin/env python3
import json
import logging
import os
import pprint as pp
import sys
from dataclasses import dataclass

import msgspec
import numpy as np
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

import pybullet_tree_sim.utils.logging_conf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# from pybullet_tree_sim.tree import Tree

__here__ = os.path.dirname(os.path.abspath(__file__))


# @dataclass
class Cylinder(msgspec.Struct):
    limb_name: str
    limb_id: int
    cylinder_id: int
    color: tuple[int, int, int]
    centroid: tuple[float, float, float]
    radius: float
    length: float
    orientation: tuple[float, float, float]
    rot_mat: np.ndarray

    def to_dict(self) -> dict:
        info_dict = {
            "limb_name": self.limb_name,
            "limb_id": self.limb_id,
            "cylinder_id": self.cylinder_id,
            "color": self.color,
            "centroid": self.centroid,
            "radius": self.radius,
            "length": self.length,
            "orientation": self.orientation,
            "rot_mat": self.rot_mat,
        }
        return info_dict


@dataclass
class Face:
    vertices: list[tuple[float, float, float]]
    normal: tuple[float, float, float]
    color: tuple[int, int, int]
    face_id: int
    face_id_a: list[int]
    theta: float
    t_val: float
    cylinder_id: int

    def to_dict(self) -> dict:
        info_dict = {
            "vertices": self.vertices,
            "normal": self.normal,
            "color": self.color,
            "id": self.face_id,
            "id_a": self.face_id_a,
            "theta": self.theta,
            "t_val": self.t_val,
            "cylinder_id": self.cylinder_id,
        }
        return info_dict


@dataclass
class Limb:
    """Logical grouping of cylinders that belong to the same limb/branch.

    Holds a list of Cylinder objects and convenience methods.
    """

    name: str
    limb_id: int
    cylinders: list[str]
    children: list[str] = None
    start_point: tuple[float, float, float] = None
    end_point: tuple[float, float, float] = None

    def get_by_id(self, limb_id: int) -> Cylinder | None:
        for c in self.cylinders:
            if c.limb_id == limb_id:
                return c
        return None

    def to_dict(self) -> dict:
        info_dict = {
            "name": self.name,
            "limb_id": self.limb_id,
            "cylinders": [cylinder.to_dict() for cylinder in self.cylinders],
            "children": self.children,
            "start_point": self.start_point,
            "end_point": self.end_point,
        }
        return info_dict


@dataclass
class TreeMetadata:
    def __init__(
        self,
        namespace: str,
        tree_id: int,
        tree_type: str,
        tree_metadata_path: str,
        # tree_mesh: Trimesh
    ) -> None:

        self.namespace = namespace
        self.tree_id = str(tree_id).zfill(5)
        self.tree_type = tree_type
        self.tree_metadata_path = tree_metadata_path

        self.raw_metadata = self._load_tree_metadata(self.tree_metadata_path)

        self.metadata = self.process_metadata(raw_metadata=self.raw_metadata)

        # self.parent_to_children = self.get_parent_to_child_map(metadata=metadata)
        self.limbs, self.cylinders = self.get_tree_geometry(metadata=self.metadata)
        self.color_map = self.get_color_to_cylinder_map()

        return

    def _load_tree_metadata(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tree metadata file not found: {file_path}")
        with open(file_path, "r") as f:
            metadata = json.load(f)
        return metadata

    def process_metadata(self, raw_metadata: dict) -> dict:
        # lower case all key names and part names in value lists. If a value is an empty list, do not include it
        hierarchy_metadata = {}
        for limb_name, child_limbs in raw_metadata["hierarchy"].items():
            key = limb_name.lower()
            if isinstance(child_limbs, list) and len(child_limbs) > 0:
                value = [v.lower() if isinstance(v, str) else v for v in child_limbs]
            hierarchy_metadata[key] = value
        raw_metadata["hierarchy"] = hierarchy_metadata
        return raw_metadata

    # def get_parent_to_child_map(self, metadata: dict):
    #     parent_to_children = {parent: children for parent, children in metadata['hierarchy'].items() if children}
    #     return parent_to_children

    def get_tree_geometry(self, metadata: dict) -> dict[str, Limb]:
        """Group cylinders by their part name.

        :param data: The cylinder data to group.
        :type data: dict
        :return: A dictionary mapping part names to Limb objects.
        :rtype: dict[str, Limb]
        """
        # Iterate over hierarchy and group cylinders by part name
        limbs: dict[str, Limb] = {}
        limb_id_counter = 0
        cylinders: list[Cylinder] = []
        cylinder_id_counter = 0
        for limb_name, cyl_data_list in metadata["hierarchy"].items():
            try:
                limb_start_point = tuple(metadata["branch_locations"][limb_name]["start"])
                limb_end_point = tuple(metadata["branch_locations"][limb_name]["end"])
            except KeyError:
                continue
            limb_cylinders = []
            # iterate over cyl_data_list and create Cylinder objects by finding matching entries in metadata['cylinder_data']
            prev_cyl_ori = None
            for cylinder_color_str, cylinder_data in metadata["cylinder_data"].items():
                try:
                    if cylinder_data["part_name"].lower() == limb_name.lower():
                        if prev_cyl_ori is None:
                            prev_cyl_ori = np.array(limb_end_point) - np.array(limb_start_point)
                            prev_cyl_ori /= np.linalg.norm(prev_cyl_ori)

                        cylinder_data["rot_mat"] = self._compute_cylinder_rotation_matrix(
                            orientation=np.array(cylinder_data["orientation"]), prev_cyl_ori=prev_cyl_ori
                        )

                        cyl_obj = Cylinder(
                            limb_name=limb_name,
                            limb_id=limb_id_counter,
                            cylinder_id=cylinder_id_counter,
                            color=tuple(map(int, cylinder_color_str.strip("()").split(","))),
                            centroid=tuple(cylinder_data["centroid"]),
                            radius=cylinder_data["radius"],
                            length=cylinder_data["length"],
                            orientation=tuple(cylinder_data["orientation"]),
                            rot_mat=cylinder_data["rot_mat"],
                        )
                        cylinder_id_counter += 1
                        limb_cylinders.append(cyl_obj)

                        prev_cyl_ori = np.array(cylinder_data["orientation"])
                except KeyError:
                    # log.debug(f"Limb '{limb_name}' was pruned.")
                    continue

            cylinders.extend(limb_cylinders)

            limbs[limb_name] = Limb(
                name=limb_name,
                limb_id=limb_id_counter,
                children=cyl_data_list,
                cylinders=limb_cylinders,
                start_point=limb_start_point,
                end_point=limb_end_point,
            )
            limb_id_counter += 1

        return limbs, cylinders

    def _compute_cylinder_rotation_matrix(self, orientation: list[float], prev_cyl_ori) -> np.ndarray:
        if np.dot(orientation, prev_cyl_ori) < 0:
            orientation = -np.array(orientation)
        world_z = np.array([0, 0, 1])
        rot_axis = np.cross(world_z, orientation)
        rot_axis /= np.linalg.norm(rot_axis)
        rot_mag = np.arccos(np.dot(world_z, orientation) / (np.linalg.norm(world_z) * np.linalg.norm(orientation)))
        rot_vec = rot_axis * rot_mag
        rot = R.from_rotvec(rot_vec)
        rot_mat = rot.as_matrix()
        return rot_mat

    def get_color_to_cylinder_map(self) -> dict[tuple[int, int, int], Cylinder]:
        color_map: dict[tuple[int, int, int], Cylinder] = {}
        for cylinder in self.cylinders:
            color_map[cylinder.color] = cylinder
        return color_map

    def get_faces(self, original_mesh: Trimesh, recolored_mesh: Trimesh) -> list[Face]:
        faces: list[Face] = []
        face_id_counter = 0

        # Basic mesh data
        verts = np.asarray(original_mesh.vertices)  # (V,3)
        faces_vertex_indices = np.asarray(original_mesh.faces, dtype=np.int64)  # (F,3)
        face_centroids = np.mean(verts[faces_vertex_indices], axis=1)  # (F,3)
        face_normals = np.asarray(original_mesh.face_normals)  # (F,3)

        # Face colors
        original_face_rgb = np.asarray(original_mesh.visual.face_colors).astype(np.int16)  # (F,4)
        recolored_face_rgb = np.asarray(recolored_mesh.visual.face_colors).astype(np.int16)  # (F,4)

        # Pack RGB into uint32 for fast comparison: R<<16 | G<<8 | B
        original_face_rgb_packed = (
            (original_face_rgb[:, 0].astype(np.uint32) << 16)
            | (original_face_rgb[:, 1].astype(np.uint32) << 8)
            | original_face_rgb[:, 2].astype(np.uint32)
        )

        # Precompute cylinder arrays
        cyl_rgb_packed = np.empty(shape=(len(self.cylinders),), dtype=np.uint32)
        cyl_centroids = np.empty(shape=(len(self.cylinders), 3), dtype=np.float32)
        cyl_ref_x = np.empty(shape=(len(self.cylinders), 3), dtype=np.float32)
        cyl_ref_y = np.empty(shape=(len(self.cylinders), 3), dtype=np.float32)
        cyl_ref_z = np.empty(shape=(len(self.cylinders), 3), dtype=np.float32)
        cyl_lengths = np.empty(shape=(len(self.cylinders),), dtype=np.float32)
        cyl_ids = np.empty(shape=(len(self.cylinders),), dtype=np.int32)

        world_x = np.array([1.0, 0.0, 0.0])
        world_y = np.array([0.0, 1.0, 0.0])
        for i, cyl in enumerate(self.cylinders):
            r, g, b = cyl.color
            packed = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)
            cyl_rgb_packed[i] = packed

            cyl_centroids[i] = np.asarray(cyl.centroid, dtype=float)
            cyl_z = np.asarray(cyl.orientation, dtype=float)
            cyl_ref_z[i] = cyl_z
            # project world_x onto plane perpendicular to axis
            cyl_x = world_x - np.dot(world_x, cyl_z) * cyl_z
            cyl_x /= np.linalg.norm(cyl_x)
            cyl_ref_x[i] = cyl_x
            cyl_y = np.cross(cyl_z, cyl_x)
            cyl_ref_y[i] = cyl_y
            cyl_lengths[i] = cyl.length
            cyl_ids[i] = cyl.cylinder_id
            # assert np.isclose(np.dot(cyl_x, cyl_z), 0.0, atol=1e-9), "Projected vector is not perpendicular to axis"

        # For each cylinder, find matching face indices by color, compute theta and t_val
        for i, cyl_rgb in enumerate(cyl_rgb_packed):
            matching_face_indices = np.where(original_face_rgb_packed == cyl_rgb)[0]
            if matching_face_indices.size == 0:
                continue
            cyl_centroid = cyl_centroids[i]
            cyl_x = cyl_ref_x[i]
            cyl_y = cyl_ref_y[i]
            cyl_z = cyl_ref_z[i]
            cyl_length = cyl_lengths[i]
            cyl_id = cyl_ids[i]

            # Vector from cylinder centroid to face centroids
            vec_centroid_to_face = face_centroids[matching_face_indices] - cyl_centroid

            # Axial coordinate (projection onto cylinder axis)
            axial_coords = np.dot(vec_centroid_to_face, cyl_z)
            t_vals = axial_coords / cyl_length

            # Radial vectors (projected onto plane perpendicular to cylinder axis)
            radial_vecs = vec_centroid_to_face - np.outer(axial_coords, cyl_z)
            radial_vecs_norm = np.linalg.norm(radial_vecs, axis=1, keepdims=True)
            radial_vecs_normalized = radial_vecs / radial_vecs_norm
            proj_x = np.dot(radial_vecs_normalized, cyl_x)
            proj_y = np.dot(radial_vecs_normalized, cyl_y)
            thetas = np.arctan2(proj_y, proj_x)

            # normals and recolored colors
            matched_normals = face_normals[matching_face_indices]
            matched_recolors = recolored_face_rgb[matching_face_indices]

            # Create Face objects
            for j, face_idx in enumerate(matching_face_indices):
                vertex_indices_for_face = faces_vertex_indices[face_idx]
                face_vertices = [tuple(original_mesh.vertices[idx]) for idx in vertex_indices_for_face]
                face = Face(
                    vertices=face_vertices,
                    normal=tuple(matched_normals[j]),
                    color=tuple(matched_recolors[j][:3]),
                    face_id=face_id_counter,
                    face_id_a=original_mesh.faces[face_idx].tolist(),
                    theta=thetas[j],
                    t_val=t_vals[j],
                    cylinder_id=cyl_id,
                )
                faces.append(face)
                face_id_counter += 1

        return faces


def main():
    # tree_meta = TreeMetadata(namespace="lpy", tree_id=0, tree_type="Envy")
    """class Cylinder(msgspec.Struct):
    limb_name: str
    limb_id: int
    cylinder_id: int
    color: tuple[int, int, int]
    centroid: tuple[float, float, float]
    radius: float
    length: float
    orientation: tuple[float, float, float]
    rot_mat: np.ndarray"""

    cyl = Cylinder(
        limb_name="trunk",
        limb_id=0,
        cylinder_id=0,
        color=(255, 0, 0),
        centroid=(0.0, 0.0, 0.0),
        radius=0.1,
        length=1.0,
        orientation=(0.0, 0.0, 0.0),
        rot_mat=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).tolist(),
    )
    print(cyl.to_dict())

    face = Face(
        vertices=[
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 3.0),
        ],
        normal=(1.0, 0.0, 0.0),
        color=(255, 0, 0),
        face_id=12,
        face_id_a=["abc123"],
        theta=np.radians(30),
        t_val=0.78,
        cylinder_id=15,
    )
    print(face.to_dict())
    return


if __name__ == "__main__":
    main()
