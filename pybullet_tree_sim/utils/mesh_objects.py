#!/usr/bin/env python3
from dataclasses import dataclass
import trimesh
import numpy as np
import os

MESHES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")


class MeshObjects:
    @staticmethod
    def load_mesh(mesh_path: str, mesh_type: str = "ply") -> trimesh.Trimesh:
        """Load a mesh from a path

        :param mesh_path: Mesh path
        :type mesh_path: str
        :return: Trimesh object of the mesh
        :rtype: trimesh.Trimesh
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Could not find file '{mesh_path}'.")
        mesh = trimesh.load_mesh(file_obj=mesh_path, file_type=mesh_type)
        return mesh

    @staticmethod
    def color_unique_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Duplicate vertices per-face so each face can get a flat color (no vertex-sharing)
        This ensures a unique set of vertices per triangle so color doesn't interpolate across adjacent faces.

        :param mesh_path: Path to the mesh file
        :type mesh_path: str
        :return: A recolored trimesh object
        :rtype: trimesh.Trimesh
        """
        faces = mesh.faces
        n_faces = faces.shape[0]
        verts_per_face = mesh.vertices[faces]  # shape (n_faces, 3, 3)
        new_vertices = verts_per_face.reshape(-1, 3)  # (n_faces*3, 3)
        new_faces = np.arange(len(new_vertices)).reshape(-1, 3)  # (n_faces, 3)

        # Encode face IDs into 24-bit RGB colors (reserve a background id 0 if you like)
        ids = np.arange(n_faces, dtype=np.uint32)
        r = (ids >> 16) & 0xFF
        g = (ids >> 8) & 0xFF
        b = ids & 0xFF
        face_colors = np.stack([r, g, b, np.full_like(r, 255)], axis=1).astype(np.uint8)  # RGBA per-face

        # Expand face colors to match new vertices
        vertex_colors = np.repeat(face_colors, 3, axis=0)  # (n_faces*3, 4)

        recolored_mesh = trimesh.Trimesh(
            vertices=new_vertices, faces=new_faces, vertex_colors=vertex_colors, process=False
        )
        return recolored_mesh

    @staticmethod
    def export_obj(
        mesh: trimesh.Trimesh,
        filename: str,
        dir_path: str = f"{MESHES_PATH}/tmp",
    ) -> None:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        export_path = os.path.join(dir_path, filename)
        mesh.export(f"{export_path}", file_type="obj")
        return

    @staticmethod
    def color_and_convert_ply_to_obj(ply_mesh_path: str, obj_mesh_path: str) -> trimesh.Trimesh:
        """Loads a mesh .obj mesh file with its path defined by the tree_id_str and tree_type, recolors it so each face has a unique color,
        and exports it to an .obj file.

        :param ply_mesh_path: Path to the .ply mesh file
        :type ply_mesh_path: str
        :param obj_mesh_path: Path to save the .obj mesh file
        :type obj_mesh_path: str
        :return: The recolored trimesh object
        :rtype: trimesh.Trimesh
        """
        ctm = MeshObjects.color_unique_faces(mesh=MeshObjects.load_mesh(ply_mesh_path))
        ctm.export(obj_mesh_path, file_type="obj")
        return ctm
