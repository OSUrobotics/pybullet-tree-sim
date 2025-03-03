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
from nptyping import NDArray, Shape, Float
from numpy.typing import ArrayLike
from pybullet_tree_sim import RGB_LABEL, URDF_PATH, MESHES_PATH, PKL_PATH
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from pybullet_tree_sim.utils.camera_helpers import (
    compute_perpendicular_projection_vector,
)
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

    _tree_xacro_path = os.path.join(URDF_PATH, "trees", "envy", "tree.urdf.xacro")
    _tree_generated_urdf_path = os.path.join(URDF_PATH, "trees", "envy", "generated")
    _tree_meshes_unlabeled_path = os.path.join(MESHES_PATH, "trees", "envy", "unlabeled", "obj")
    _tree_meshes_labeled_path = os.path.join(MESHES_PATH, "trees", "envy", "labeled", "obj")
    _pkl_path = PKL_PATH

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
        self.generator: np.random.Generator = np.random.default_rng(seed=self.seed)
        self.verbose = verbose

        # Tree specific parameters
        self.scale = scale
        self.tree_namespace = namespace
        self.tree_id = tree_id
        self.tree_type = tree_type
        self.id_str = self.create_id_string(tree_id=tree_id, tree_type=tree_type, namespace=namespace, urdf_path=urdf_path)
        self.urdf_path = os.path.join(self._tree_generated_urdf_path, self.id_str+'.urdf')
        self.mesh_path = os.path.join(self._tree_meshes_unlabeled_path, self.id_str+".obj")
        self.labeled_mesh_path = os.path.join(self._tree_meshes_labeled_path, self.id_str+"_labeled.obj")
        self.init_pos = position
        self.init_orientation = orientation


        log.info(f"__init__ {self.id_str}")


        # URDF
        self.load_tree_urdf(scale=scale, parent=parent)
        # OBJ
        tree_obj = self.load_tree_obj()
        log.info(f"Tree mesh loaded: {self.mesh_path}")
        # Labelled OBJ
        labeled_tree_obj = self.load_labeled_tree_obj()

        # Tree specific parameters
        self.rgb_label = RGB_LABEL
        self.pyb_tree_id = None



        # Set tree pose
        if randomize_pose:
            new_pos, new_orientation = self._randomize_pose()
        else:
            new_pos = position
            new_orientation = orientation

        self.pos = new_pos
        self.orientation = new_orientation

        # # Variables to store the vertices and statistics of the tree
        self.vertex_and_projection = []
        self.projection_mean = np.array(0.0)
        self.projection_std = np.array(0.0)
        self.projection_sum_x = np.array(0.0)
        self.projection_sum_x2 = np.array(0.0)
        self.reachable_points = []

        # Label textured tree
        vertex_to_label = self.label_vertex_by_color(self.rgb_label, tree_obj.vertices, labeled_tree_obj.vertices)

        # append the label to each vertex
        tree_obj_vertices_labeled = []
        for i, vertex in enumerate(tree_obj.vertices):
            tree_obj_vertices_labeled.append(vertex + (vertex_to_label[vertex],))

        # # # TODO: begin() method or something
        self.transformed_vertices = list(
            map(
                lambda x: self.transform_tree_obj_vertex(x),
                tree_obj_vertices_labeled,
            )
        )

        # if pickled file exists load and return
        path_component = os.path.normpath(self.urdf_path).split(os.path.sep)
        # log.info(f"PATH COMPONENT: {path_component}")

        pkl_filepath = os.path.join(self._pkl_path, str(path_component[-1][:-5]) + "_points.pkl")
        if os.path.exists(pkl_filepath):
            self._load_points_from_pickle(pkl_filepath)
        else:
            # Get all points on the tree
            self.get_all_points(tree_obj)
            self.filter_outliers()
            self.filter_trunk_points()
            # self.filter_points_below_base()

            # dump reachable points to file using pickle
            with open(pkl_filepath, "wb") as f:
                pickle.dump((self.pos, self.orientation, self.vertex_and_projection), f)

            if self.verbose > 0:
                print(f"INFO: Number of points: {len(self.vertex_and_projection)}")
                print(f"INFO: Saved points to pickle file {pkl_filepath}")

        # # Make different meshes for each label
        # # make bins
        self.or_bins = self.create_bins(18, 36)
        # # Go through vertex_and_projection and assign to bins
        self.populate_bins(self.vertex_and_projection)

        del self.vertex_and_projection

        return

    def create_id_string(self,
        tree_id: int | None = None,
        tree_type: str | None = None,
        namespace: str | None = None,
        urdf_path: str | None = None
    ) -> str:
        if tree_id is None and urdf_path is None:
            # log.error("Both urdf_path and tree parameters cannot be None.")
            raise TreeException("Both urdf_path and tree parameters cannot be None.")

        if urdf_path is None:
            id_str = f"{namespace}_{tree_type}_tree{tree_id}"
        else:
            id_str = Path(urdf_path).stem
            id_str_components = id_str.split('_')
            self.tree_namespace = id_str_components[0]
            self.tree_type = id_str_components[1]
            self.tree_id = id_str_components[2]
        return id_str

    def _load_points_from_pickle(self, pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.pos = data[0]
            self.orientation = data[1]
            self.vertex_and_projection = data[2]
        if self.verbose > 0:
            log.info(f"Loaded points from pickle file {pkl_path}")
            log.info(f"Number of points: {len(self.vertex_and_projection)}")
        return

    def _randomize_pose(self) -> tuple:
        # TODO: Randomize position to bounds?
        new_position = np.array([0, 0, 0])
        # TODO: Multiply orientation with initial orientation
        new_orientation = pybullet.getQuaternionFromEuler(
            self.generator.uniform(low=-1, high=1, size=(3,)) * np.pi / 180 * 5
        )
        return new_position, new_orientation

    @staticmethod
    def create_bins(num_latitude_bins, num_longitude_bins):
        """
        Create bins separated by 10 degrees on a unit sphere.

        Parameters:
            num_latitude_bins (int): Number of bins along the latitude direction.
            num_longitude_bins (int): Number of bins along the longitude direction.

        Returns:
            list of tuples: List of tuples where each tuple represents a bin defined by
                            (latitude_min, latitude_max, longitude_min, longitude_max).
        """
        bin_size = np.deg2rad(10)  # Convert degrees to radians
        bins = {}
        for i in range(num_latitude_bins):
            lat_min = np.rad2deg(-np.pi / 2 + i * bin_size)
            lat_max = np.rad2deg(-np.pi / 2 + (i + 1) * bin_size)
            for j in range(num_longitude_bins):
                lon_min = np.rad2deg(-np.pi + j * bin_size)
                lon_max = np.rad2deg(-np.pi + (j + 1) * bin_size)
                bins[
                    (
                        round((lat_min + lat_max) / 2),
                        round((lon_min + lon_max) / 2),
                    )
                ] = []
        return bins

    def transform_tree_obj_vertex(self, vertex: ArrayLike) -> Tuple[np.ndarray, float]:
        """
        Transform a vertex from the tree object to the world frame.
        """
        vertex_pos = np.array(vertex[0:3]) * self.scale
        vertex_orientation = [0, 0, 0, 1]  # Dont care about orientation

        vertex_w_transform: Tuple[tuple, tuple] = self.pbclient.multiplyTransforms(
            self.pos, self.orientation, vertex_pos, vertex_orientation
        )
        # vertex_w_transform = np.concatenate((final_position, final_orientation))

        return (np.array(vertex_w_transform[0]), vertex[3])

    def populate_bins(self, points):
        """
        Populate the bins based on a list of direction vectors.

        Parameters:
            direction_vectors (list of numpy.ndarray): List of direction vectors.
            bins (list of tuples): List of bins where each tuple represents a bin defined by
                                   (latitude_min, latitude_max, longitude_min, longitude_max).

        Returns:
            list of lists: List of lists where each sublist represents the indices of direction vectors
                           assigned to the corresponding bin.
        """
        offset = 1e-3
        for i, point in enumerate(points):

            direction_vector = point[1]
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            lat_angle = np.rad2deg(np.arcsin(direction_vector[2])) + offset
            lon_angle = np.rad2deg(np.arctan2(direction_vector[1], direction_vector[0])) + offset
            lat_angle_min = mh.rounddown(lat_angle)
            lat_angle_max = mh.roundup(lat_angle)
            lon_angle_min = mh.rounddown(lon_angle)
            lon_angle_max = mh.roundup(lon_angle)
            bin_key = (
                round((lat_angle_min + lat_angle_max) / 2),
                round((lon_angle_min + lon_angle_max) / 2),
            )
            # if bin_key[0] not in between -85 and 85 set as 85 or -85
            # if bin_keyp[1] not in between -175 and 175 set as 175 or -175

            if bin_key[0] > 85:
                bin_key = (85, bin_key[1])
            elif bin_key[0] < -85:
                bin_key = (-85, bin_key[1])
            if bin_key[1] > 175:
                bin_key = (bin_key[0], 175)
            elif bin_key[1] < -175:
                bin_key = (bin_key[0], -175)
            self.or_bins[bin_key].append(
                (self.urdf_path, point, self.orientation, self.scale)
            )  # Add collision meshes for each tree here
        return

    def label_vertex_by_color(self, labels, unlabelled_vertices, labelled_vertices):
        # create a dictionary of vertices and assign label using close enough vertex on labelled tree obj
        vertex_to_label = {}
        for i, vertex in enumerate(unlabelled_vertices):
            vertex_to_label[vertex] = None

        for j, labelled_vertex in enumerate(labelled_vertices):
            min_dist = 100000
            for i in labels.keys():
                # assign label that is closest
                dist = np.linalg.norm(np.array(labelled_vertex[3:]) - np.array(i))
                if dist < min_dist:
                    min_dist = dist
                    vertex_to_label[labelled_vertex[:3]] = labels[i]
        return vertex_to_label

    def get_all_points(self, tree_obj):
        for num, face in enumerate(tree_obj.mesh_list[0].faces):
            # Order the sides of the face by length
            ab = (
                face[0],
                face[1],
                np.linalg.norm(self.transformed_vertices[face[0]][0] - self.transformed_vertices[face[1]][0]),
            )
            ac = (
                face[0],
                face[2],
                np.linalg.norm(self.transformed_vertices[face[0]][0] - self.transformed_vertices[face[2]][0]),
            )
            bc = (
                face[1],
                face[2],
                np.linalg.norm(self.transformed_vertices[face[1]][0] - self.transformed_vertices[face[2]][0]),
            )

            normal_vec = np.cross(
                self.transformed_vertices[ac[0]][0] - self.transformed_vertices[ac[1]][0],
                self.transformed_vertices[bc[0]][0] - self.transformed_vertices[bc[1]][0],
            )
            # Only front facing faces
            if np.dot(normal_vec, [0, 1, 0]) < 0:
                continue
            # argsort sorts in ascending order
            sides = [ab, ac, bc]
            sorted_sides = np.argsort([x[2] for x in sides])
            ac = sides[sorted_sides[2]]
            ab = sides[sorted_sides[1]]
            bc = sides[sorted_sides[0]]
            # |a
            # |\
            # | \
            # |  \
            # |   \
            # |    \
            # b______\c
            perpendicular_projection = compute_perpendicular_projection_vector(
                self.transformed_vertices[ac[0]][0] - self.transformed_vertices[ac[1]][0],
                self.transformed_vertices[bc[0]][0] - self.transformed_vertices[bc[1]][0],
            )

            scale = np.random.uniform()
            tree_point = (1 - scale) * self.transformed_vertices[ab[0]][0] + scale * self.transformed_vertices[ab[1]][0]

            # Label the face as the majority label of the verticesp
            labels = [
                self.transformed_vertices[ab[0]][1],
                self.transformed_vertices[ab[1]][1],
                self.transformed_vertices[ac[0]][1],
                self.transformed_vertices[ac[1]][1],
                self.transformed_vertices[bc[0]][1],
                self.transformed_vertices[bc[1]][1],
            ]

            # If all three vertices are the same label, assign that label
            # else assign label "JOINT"
            if len(set(labels)) == 1:
                label = labels[0]
            else:
                label = "JOINT"
            if label != "SPUR" and label != "WATER_BRANCH":
                continue
            self.vertex_and_projection.append((tree_point, perpendicular_projection, normal_vec, label))
            # This projection mean is used to filter corner/flushed faces which do not correspond to a branch
            self.projection_sum_x += np.linalg.norm(perpendicular_projection)
            self.projection_sum_x2 += np.linalg.norm(perpendicular_projection) ** 2

        self.projection_mean = self.projection_sum_x / len(self.vertex_and_projection)
        self.projection_std = np.sqrt(
            self.projection_sum_x2 / len(self.vertex_and_projection) - self.projection_mean**2
        )
        return

    def filter_outliers(self):
        # Filter out outliers
        print(
            "Number of points before filtering: ",
            len(self.vertex_and_projection),
        )
        self.vertex_and_projection = list(
            filter(
                lambda x: np.linalg.norm(x[1]) > self.projection_mean + 0.5 * self.projection_std,
                self.vertex_and_projection,
            )
        )
        print(
            "Number of points after filtering: ",
            len(self.vertex_and_projection),
        )
        return

    def filter_points_below_base(self, base_xyz):
        # Filter out points below the base of the arm
        self.vertex_and_projection = list(filter(lambda x: x[0][2] > base_xyz[2], self.vertex_and_projection))
        return

    def filter_trunk_points(self):
        self.vertex_and_projection = list(
            filter(
                lambda x: abs(x[0][0] - self.pos[0]) > 0.8,
                self.vertex_and_projection,
            )
        )
        print(
            "Number of points after filtering trunk points: ",
            len(self.vertex_and_projection),
        )
        return

    def load_tree_urdf(
        self,
        scale: float,
        parent: str = "world",
        position: str = "0.0 0.0 0.0",
        orientation: str = "0.0 0.0 0.0",
        save_urdf: bool = True,
        regenerate_urdf: bool = False,  # TODO: make save/regenerate work well together. Will need to add delete URDF function
    ) -> None:
        """Load a tree URDF from a given path or generate a tree URDF from a xacro file. Returns the URDF content.
        If `tree_urdf_path` is not None, then load that URDF.
        Otherwise, process an xacro file with given input parameters.

        Returns
        -------
            None
        """
        if not os.path.exists(self.urdf_path):
            log.info(f"Could not find file '{self.urdf_path}'. Generating URDF from xacro.")

            if not os.path.isdir(Tree._tree_generated_urdf_path):
                os.mkdir(Tree._tree_generated_urdf_path)

                urdf_mappings = {
                    "namespace": self.tree_namespace,
                    "tree_id": str(self.tree_id),
                    "tree_type": self.tree_type,
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
                    log.info(f"Saved URDF to file '{self.urdf_path}'.")
        else:
            urdf_content = xutils.load_urdf_from_xacro(xacro_path=self.urdf_path).toprettyxml()
            log.info(f"Loaded URDF from file '{self.urdf_path}'.")

        return

    def load_tree_obj(self):
        if not os.path.exists(self.mesh_path):
            raise TreeException(f"Could not find file '{self.mesh_path}.")
        tree_obj = pywavefront.Wavefront(self.mesh_path, create_materials=True, collect_faces=True)
        return tree_obj

    def load_labeled_tree_obj(self):
        if not os.path.exists(self.labeled_mesh_path):
            raise TreeException(f"Could not find the file {self.labeled_mesh_path}")
        labeled_tree_obj = pywavefront.Wavefront(self.labeled_mesh_path, create_materials=True, collect_faces=True)
        return labeled_tree_obj

    @staticmethod
    def make_trees_from_ids(
        pbutils: PyBUtils,
        tree_ids: list[int],
        namespace: str = '',
        pos: np.ndarray = np.array([0,0,0]),
        orientation: np.ndarray = np.array([0,0,0,1]),
        scale: float = 1.0,
        randomize_pose: bool = False
    ) -> list[Tree]:
        trees: list[Tree] = []

        for tree_id in tree_ids:
            trees.append(
                Tree(
                    pbutils=pbutils,
                    tree_id=tree_id,

                )
            )
        return trees

    @staticmethod
    def make_trees_from_folder(
        pbutils: PyBUtils,
        trees_urdf_path: str,
        trees_obj_path: str,
        trees_labelled_path: str,
        pos: np.ndarray,
        orientation: np.ndarray,
        scale: float,
        num_trees: int,
        randomize_pose: bool = False,
    ) -> list[Tree]:
        trees: list[Tree] = []
        # for urdfs, objs in zip(
        #     sorted(glob.glob(trees_urdf_path + "/*.urdf")), sorted(glob.glob(trees_obj_path + "/*.obj"))
        # ):
        #     print(urdfs, objs, labelled_obj)
        #     if len(trees) >= num_trees:
        #         break
        #     trees.append(
        #         Tree(
        #             urdf_path=urdfs,
        #             obj_path=objs,
        #             pos=pos,
        #             orientation=orientation,
        #             scale=scale,
        #             randomize_pose=randomize_pose,
        #         )
        #     )

        # self,
        # pbutils: PyBUtils,
        # tree_id: int | None = None,
        # tree_type: str | None = None,
        # namespace: str = "",
        # parent: str = "world",
        # urdf_path: str | None = None,
        # obj_path: str | None = None,
        # labeled_tree_obj_path: str | None = None,
        # position=np.array([0, 0, 0]),
        # orientation=np.array([0, 0, 0, 1]),
        # scale: float = 1.0,
        # randomize_pose: bool = False,
        # verbose: bool = True,
        # seed: int | None = None,
        return trees


def main():

    return


if __name__ == "__main__":
    main()
