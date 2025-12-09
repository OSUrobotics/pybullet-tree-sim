#!/usr/bin/env python3
from dataclasses import dataclass
import json
import os
import pprint as pp
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R
from trimesh import Trimesh

import msgspec

from zenlog import log

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


class Face(msgspec.Struct):
    vertices: list[tuple[float, float, float]]
    normal: tuple[float, float, float]
    color: tuple[int, int, int]
    face_id: int
    face_id_a: list[int]
    theta: float
    t_val: float
    cylinder_id: int


class Limb(msgspec.Struct):
    """Logical grouping of cylinders that belong to the same limb/branch.

    Holds a list of Cylinder objects and convenience methods.
    """

    name: str
    limb_id: int
    cylinders: list[str]
    children: list[str] = None

    def get_by_id(self, limb_id: int) -> Cylinder | None:
        for c in self.cylinders:
            if c.limb_id == limb_id:
                return c
        return None


@dataclass
class TreeMetadata:
    def __init__(
        self, namespace: str, tree_id: int, tree_type: str, tree_metadata_path: str, tree_mesh: Trimesh
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

            limb_cylinders = []
            # iterate over cyl_data_list and create Cylinder objects by finding matching entries in metadata['cylinder_data']
            for cylinder_color_str, cylinder_data in metadata["cylinder_data"].items():
                try:
                    if cylinder_data["part_name"].lower() == limb_name.lower():
                        cyl_obj = Cylinder(
                            limb_name=limb_name,
                            limb_id=limb_id_counter,
                            cylinder_id=cylinder_id_counter,
                            color=tuple(map(int, cylinder_color_str.strip("()").split(","))),
                            centroid=tuple(cylinder_data["centroid"]),
                            radius=cylinder_data["radius"],
                            length=cylinder_data["length"],
                            orientation=tuple(cylinder_data["orientation"]),
                        )
                        cylinder_id_counter += 1
                        limb_cylinders.append(cyl_obj)
                except KeyError:
                    # log.debug(f"Limb '{limb_name}' was pruned.")
                    continue

            cylinders.extend(limb_cylinders)

            limbs[limb_name] = Limb(
                name=limb_name, limb_id=limb_id_counter, children=cyl_data_list, cylinders=limb_cylinders
            )
            limb_id_counter += 1

        return limbs, cylinders

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
        face_idx = np.asarray(original_mesh.faces, dtype=np.int64)  # (F,3)
        face_centroids = np.mean(verts[face_idx], axis=1)  # (F,3)
        face_normals = np.asarray(original_mesh.face_normals)  # (F,3)

        # Face colors
        original_face_colors = np.asarray(original_mesh.visual.face_colors).astype(np.int16)  # (F,4)
        recolored_face_colors = np.asarray(recolored_mesh.visual.face_colors).astype(np.int16)  # (F,4)
        
        # Pack RGB into uint32 for fast comparison: R<<16 | G<<8 | B
        face_packed = (
            (original_face_colors[:, 0].astype(np.uint32) << 16)
            | (original_face_colors[:, 1].astype(np.uint32) << 8)
            | original_face_colors[:, 2].astype(np.uint32)
        )

        print(face_packed)

        import sys

        sys.exit(0)

        # Iterate over mesh faces
        for face_idx in range(len(original_mesh.faces)):
            vertex_indices = original_mesh.faces[face_idx]
            vertices = [tuple(original_mesh.vertices[idx]) for idx in vertex_indices]
            normal = tuple(original_mesh.face_normals[face_idx])

            original_color = tuple(original_mesh.visual.face_colors[face_idx][:3])  # RGB only
            recolored_color = tuple(recolored_mesh.visual.face_colors[face_idx][:3])  # RGB only
            # print(original_color, recolored_color)
            cylinder_part = self.color_map.get(original_color, None)
            if cylinder_part is None:
                # Check cylinder metadata to confirm it's a pruned part
                check_cyl = self.raw_metadata["cylinder_data"].get(str(original_color), None)
                if check_cyl is None:
                    log.warning(f"Original face color {original_color} not found in cylinder color map.")
                else:
                    pass
                    # log.debug(f"Original face color {original_color} corresponds to a pruned cylinder part.")
                continue

            cylinder_part_name = cylinder_part.limb_name
            cylinder_part_id = cylinder_part.limb_id

            # Compute theta and t_val. Theta is the angle around the cylinder axis, t_val is the normalized height along the cylinder. We assume the cylinder axis is aligned with the z-axis for simplicity.
            face_center = np.mean(vertices, axis=0)
            t_val = (face_center[2] - cylinder_part.centroid[2]) / cylinder_part.length

            # Get the phi angle which includes the centroid and orientation of the cylinder

            x_axis = np.array([1, 0, 0])
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, cylinder_part.orientation)
            rot_axis_norm = np.linalg.norm(rot_axis)
            if rot_axis_norm != 0:
                rot_axis = rot_axis / rot_axis_norm
                phi = np.arccos(np.dot(z_axis, cylinder_part.orientation))  # Angle between z-axis and cylinder
            else:
                phi = 0.0
            rot = R.from_rotvec(rot_axis * phi)
            rotated_x = rot.apply(x_axis)
            # find angle between rotated_x and face normal
            theta = np.arccos(np.dot(rotated_x, normal) / (np.linalg.norm(rotated_x) * np.linalg.norm(normal)))
            if np.isnan(theta):
                theta = 0.0

            face = Face(
                vertices=vertices,
                normal=normal,
                color=recolored_color,
                face_id=face_id_counter,
                face_id_a=original_mesh.faces[face_idx],
                theta=theta,
                t_val=t_val,
                cylinder_id=cylinder_part_id,
            )
            faces.append(face)
            face_id_counter += 1
            self.faces = faces
        return faces


def main():
    # tree_meta = TreeMetadata(namespace="lpy", tree_id=0, tree_type="Envy")

    cyl = Cylinder(
        part_name="trunk",
        part_id=0,
        color=(255, 0, 0),
        centroid=(0.0, 0.0, 0.0),
        radius=0.1,
        length=1.0,
        orientation=(0.0, 0.0, 0.0),
    )
    # print(cyl)
    msg = msgspec.json.encode(cyl)
    # print(msg)
    return


if __name__ == "__main__":
    main()
