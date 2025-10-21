#!/usr/bin/env python3
from __future__ import annotations

"""
tree.py
authors: Abhinav Jain, Luke Strohbehn

Generates a tree in PyBullet
"""
from collections import defaultdict
import glob
import math
import os
from pathlib import Path
import pickle
from typing import Optional, Tuple, List
import secrets
import numpy as np
import pybullet
import pywavefront

# from nptyping import NDArray, Shape, Float
from numpy.typing import ArrayLike
from pybullet_tree_sim import URDF_PATH, MESHES_PATH
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.camera_helpers import (
    compute_perpendicular_projection_vector,
)
import pybullet_tree_sim.utils.trimesh_render as tmr
from pybullet_tree_sim.utils import math_helpers as mh
import pybullet_tree_sim.utils.xacro_utils as xutils
from scipy.spatial.transform import Rotation
import xacro
import xml

from zenlog import log


# from pruning_sb3.pruning_gym.helpers import roundup, rounddown
# from memory_profiler import profile


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
    _tree_meshes_ply_path = os.path.join(MESHES_PATH, "trees")
    _tree_meshes_obj_path = os.path.join(MESHES_PATH, "trees", "obj")

    def __init__(
        self,
        pbutils: PyBUtils,
        tree_id: int | None = None,
        tree_type: str | None = None,
        namespace: str = "",
        parent: str = "world",
        urdf_path: str | None = None,
        obj_path: str | None = None,
        labeled_tree_obj_path: str | None = None,
        position: np.ndarray = np.array([0, 0, 0]),
        orientation: np.ndarray = np.array([0, 0, 0, 1]),
        scale: float = 1.0,
        randomize_pose: bool = False,
        verbose: bool = True,
        seed: int | None = None,
    ) -> None:
        log.info("Creating Tree object")
        self.pbclient = pbutils.pbclient
        # Set seed
        if seed is not None:
            self.seed = seed
        else:
            self.seed = secrets.randbits(128)
        self.generator: np.random.Generator = np.random.default_rng(
            seed=self.seed
        )
        self.verbose = verbose

        # Tree specific parameters
        self.scale = scale
        self.tree_namespace = namespace
        self.tree_id = tree_id
        self.tree_type = tree_type
        self.id_str = self.create_id_string(
            tree_id=tree_id,
            tree_type=tree_type,
            namespace=namespace,
            urdf_path=urdf_path,
        )
        self.urdf_path = os.path.join(
            self._tree_generated_urdf_path, self.id_str + ".urdf"
        )
        self.ply_mesh_path = os.path.join(
            self._tree_meshes_ply_path, self.id_str + ".ply"
        )
        self.obj_mesh_path = os.path.join(
            self._tree_meshes_obj_path, self.id_str + ".obj"
        )
        self.init_pos = position
        self.init_orientation = orientation

        # Trimesh renderer
        self.tm_renderer = None
        # OBJ
        self.convert_tree_ply_to_obj()
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

    def create_id_string(
        self,
        tree_id: int | None = None,
        tree_type: str | None = None,
        namespace: str | None = None,
        urdf_path: str | None = None,
    ) -> str:
        if tree_id is None and urdf_path is None:
            # log.error("Both urdf_path and tree parameters cannot be None.")
            raise TreeException(
                "Both urdf_path and tree parameters cannot be None."
            )

        if tree_id is not None:
            tree_id = str(tree_id).zfill(5)

        if urdf_path is None:
            id_str = f"{namespace}_{tree_type}_{tree_id}"
        else:
            id_str = Path(urdf_path).stem
            id_str_components = id_str.split("_")
            self.tree_namespace = id_str_components[0]
            self.tree_type = id_str_components[1]
            self.tree_id = str(id_str_components[2]).zfill(5)
        return id_str

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
            log.info(
                f"Could not find file '{self.urdf_path}'. Generating URDF from xacro."
            )

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
                xutils.save_urdf(
                    urdf_content=urdf_content, urdf_path=self.urdf_path
                )
        else:
            urdf_content = xutils.load_urdf_from_xacro(
                xacro_path=self.urdf_path
            ).toprettyxml()
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

    def transform_tree_obj_vertex(
        self, vertex: ArrayLike
    ) -> Tuple[np.ndarray, float]:
        """
        Transform a vertex from the tree object to the world frame.
        """
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation

        vertex_w_transform: Tuple[tuple, tuple] = (
            self.pbclient.multiplyTransforms(
                self.pos, self.orientation, vertex_pos, vertex_orientation
            )
        )
        # vertex_w_transform = np.concatenate((final_position, final_orientation))

        return (np.array(vertex_w_transform[0]), vertex[3])

    def convert_tree_ply_to_obj(self) -> None:
        """Loads a mesh .obj mesh file with its path defined by the tree_id_str"""
        if not os.path.exists(self.ply_mesh_path):
            raise TreeException(f"Could not find file '{self.ply_mesh_path}.")
        tm = tmr.RenderScene.load_mesh(mesh_path=self.ply_mesh_path)
        ctm = tmr.RenderScene.recolor_mesh(mesh=tm)
        ctm.export(self.obj_mesh_path, file_type="obj")
        return


def main():
    import time

    from pybullet_tree_sim.utils.pyb_utils import PyBUtils

    pbutils = PyBUtils(renders=True)

    tree = Tree(
        pbutils=pbutils,
        tree_id=64,
        tree_type="envy",
        namespace="LPy",
    )
    sim_start_time = time.time()
    while time.time() - sim_start_time < 10:
        continue

    return


if __name__ == "__main__":
    main()
