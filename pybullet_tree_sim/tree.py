#!/usr/bin/env python3
from __future__ import annotations

"""
tree.py
authors: Abhinav Jain, Luke Strohbehn

Generates a tree in PyBullet
"""
from collections import defaultdict
import os
from pathlib import Path
import secrets
import numpy as np
import pybullet

from numpy.typing import ArrayLike
from pybullet_tree_sim import URDF_PATH, MESHES_PATH
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.mesh_objects import MeshObjects
import pybullet_tree_sim.utils.xacro_utils as xutils
from pybullet_tree_sim.tree_metadata import TreeMetadata
import tempfile

from zenlog import log


class TreeException(Exception):
    pass


class Tree:
    """This class is used to create a tree object by loading the urdf file and the obj file
    along with the labelled obj file. This class is used to filter the points on the tree # and create a curriculum of points
    to be used in training.

    To create a tree object, the following parameters are required:
    urdf_path: The path to the urdf file
    obj_path: The path to the obj file
    labelled_obj_path: The path to the labelled obj file
    pos: The position of the tree
    orientation: The orientation of the tree
    scale: The scale of the tree
    """

    _tree_xacro_path = os.path.join(URDF_PATH, "tree", "tree.urdf.xacro")
    _tree_generated_urdf_path = os.path.join(URDF_PATH, "tree", "generated")

    def __init__(
        self,
        pbutils: PyBUtils,
        meshes_root: str,
        tree_id: int,
        tree_type: str,
        namespace: str = "",
        parent: str = "world",
        position: np.ndarray = np.array([0, 0, 0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        scale: float = 1.0,
        randomize_pose: bool = False,
        seed: int | None = None,
    ) -> None:
        # Set up temporary directory for tree files
        self.temp_dir = tempfile.TemporaryDirectory()
        log.info(f"Creating Tree object '{namespace}_{tree_type}_{tree_id}'.")

        # PyBullet
        self.pbclient = pbutils.pbclient

        # Set seed
        if seed is not None:  # TODO: Log this seed for reproducibility
            self.seed = seed
        else:
            self.seed = secrets.randbits(128)
        self.generator: np.random.Generator = np.random.default_rng(seed=self.seed)

        # Tree specific parameters
        self.scale = scale
        self.tree_namespace = namespace.lower().strip()
        self.tree_id = str(tree_id).zfill(5)
        self.tree_type = tree_type.lower().strip()
        self.id_str = f"{self.tree_namespace}_{self.tree_type}_{self.tree_id}"

        # set up paths
        self._tree_meshes_ply_path = os.path.join(meshes_root, "trees", "ply")
        self._tree_meshes_obj_path = os.path.join(meshes_root, "trees", "obj")
        self._tree_meshes_metadata_path = os.path.join(meshes_root, "trees", "metadata")
        self.urdf_path = os.path.join(self._tree_generated_urdf_path, self.id_str + ".urdf")
        self.ply_mesh_path = os.path.join(self._tree_meshes_ply_path, self.id_str + ".ply")
        self.obj_mesh_path = os.path.join(self._tree_meshes_obj_path, self.id_str + ".obj")
        self.mesh_metadata_path = os.path.join(self._tree_meshes_metadata_path, self.id_str + "_metadata.json")

        # Original raw mesh
        self.raw_mesh = MeshObjects.load_mesh(self.ply_mesh_path)

        # Tree metadata
        tree_metadata = TreeMetadata(
            tree_id=self.tree_id,
            tree_type=self.tree_type,
            namespace=self.tree_namespace,
            tree_metadata_path=self.mesh_metadata_path,
            tree_mesh=self.raw_mesh,
        )
        self.limbs = tree_metadata.limbs
        self.cylinders = tree_metadata.cylinders

        # Mesh with unique face colors
        self.recolored_mesh = MeshObjects.color_and_convert_ply_to_obj(
            ply_mesh_path=self.ply_mesh_path, obj_mesh_path=self.obj_mesh_path
        )
        import time

        start_time = time.process_time()
        self.faces = tree_metadata.get_faces(original_mesh=self.raw_mesh, recolored_mesh=self.recolored_mesh)
        print(f"time: {time.process_time() - start_time}")
        print(self.faces[64])
        import sys

        sys.exit(0)

        # URDF
        self.load_tree_urdf(scale=scale, parent=parent)

        # PyBullet parameters
        self.pyb_id: int = None

        # Set tree pose
        if randomize_pose:
            new_pos, new_orientation = self._randomize_pose()
        else:
            new_pos = position
            new_orientation = orientation
        self.pos = new_pos
        self.orientation = new_orientation
        return

    def load_tree_urdf(
        self,
        scale: float,
        parent: str = "world",
        position: str = "0.0 0.0 0.0",
        orientation: str = "0.0 0.0 0.0",
        save_urdf: bool = True,
        regenerate_urdf: bool = False,  # TODO: make save/regenerate work well together. Will need to add delete URDF function
    ) -> str:
        """Load a tree URDF from a given path or generate a tree URDF from a xacro file. If content is generated, by default saves the content to /urdf/trees/<tree_type>/generated Returns the URDF content.

        Returns
        -------
            None
        """
        if not os.path.exists(self.urdf_path):
            log.info(f"Could not find file '{self.urdf_path}'. Generating URDF from xacro.")

            if not os.path.isdir(Tree._tree_generated_urdf_path):
                os.mkdir(Tree._tree_generated_urdf_path)

            urdf_mappings = {
                "tree_name": self.id_str,
                "parent": parent,
                "xyz": position,
                "rpy": orientation,
            }

            # If the tree macro information doesn't describe a generated file, generate it using the generic tree xacro.
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=Tree._tree_xacro_path, mappings=urdf_mappings
            ).toprettyxml()
            if save_urdf:
                xutils.save_urdf(urdf_content=urdf_content, urdf_path=self.urdf_path)
        else:
            urdf_content = xutils.load_urdf_from_xacro(xacro_path=self.urdf_path).toprettyxml()
            log.info(f"Loaded URDF from file '{self.urdf_path}'.")

        return urdf_content

    def _randomize_pose(self) -> tuple:
        # TODO: Randomize position to bounds?
        new_position = np.array([0, 0, 0])
        # TODO: Multiply orientation with initial orientation
        new_orientation = pybullet.getQuaternionFromEuler(
            self.generator.uniform(low=-1, high=1, size=(3,)) * np.pi / 180 * 5
        )
        return new_position, new_orientation

    def transform_tree_obj_vertex(self, vertex: ArrayLike) -> tuple[np.ndarray, float]:
        """
        Transform a vertex from the tree object to the world frame.
        """
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation

        vertex_w_transform: tuple[tuple, tuple] = self.pbclient.multiplyTransforms(
            self.pos, self.orientation, vertex_pos, vertex_orientation
        )
        # vertex_w_transform = np.concatenate((final_position, final_orientation))

        return (np.array(vertex_w_transform[0]), vertex[3])


def main():
    import time

    from pybullet_tree_sim.utils.pyb_utils import PyBUtils

    pbutils = PyBUtils(renders=False)

    tree = Tree(
        pbutils=pbutils,
        meshes_root="/home/luke/dev/pybullet/pybullet-tree-sim/pybullet_tree_sim/meshes",
        tree_id=1,
        tree_type="envy",
        namespace="LPy",
    )
    sim_start_time = time.time()
    while time.time() - sim_start_time < 10:
        continue

    return


if __name__ == "__main__":
    main()
