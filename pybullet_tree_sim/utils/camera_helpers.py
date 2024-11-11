#!/usr/bin/env python3
from __future__ import annotations
from pybullet_tree_sim.utils.pyb_utils import PyBUtils
from nptyping import NDArray, Shape, Float
from typing import Union
import numpy as np


def compute_perpendicular_projection_vector(ab: NDArray[Shape["3, 1"], Float], bc: NDArray[Shape["3, 1"], Float]):
    projection = ab - np.dot(ab, bc) / np.dot(bc, bc) * bc
    return projection


def get_dfov_from_fov(fov: tuple):
    return


def get_fov_from_dfov(camera_width: int, camera_height: int, dFoV: Union[int, float], degrees: bool = True):
    """
    Returns the vertical and horizontal field of view (FoV) in degrees given the diagonal field of view (dFoV) in degrees.
    https://www.litchiutilities.com/docs/fov.php
    https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html

    @param camera_width (int): pixel width of the camera image
    @param camera_height (int): pixel height of the camera image
    @param dFoV (int/float): diagonal field of view of the camera (degrees).
    """
    for key, val in locals().items():
        if key == "degrees":
            pass
        if val <= 0:
            raise ValueError(f"Parameter '{key}' cannot be less than 0. Value: {val}")
    if degrees:
        _dfov = np.deg2rad(dFoV)
    else:
        _dfov = dFoV
    camera_diag = np.sqrt(camera_width ** 2 + camera_height ** 2)
    fov_h = 2 * np.arctan(np.tan(_dfov / 2) * camera_height / camera_diag)
    fov_w = 2 * np.arctan(np.tan(_dfov / 2) * camera_width / camera_diag)
    return (np.rad2deg(fov_w), np.rad2deg(fov_h))


# def get_pyb_proj_mat(vfov: float, aspect: float, nearVal: float, farVal: float):
#     return pbutils.pbclient.computeProjectionMatrixFOV(
#         fov=vfov, aspect=(self.depth_width / self.depth_height), nearVal=near_val, farVal=far_val
#     )

if __name__ == "__main__":
    camera_width = 64
    camera_height = 64
    dfov = 65

    fov = get_fov_from_dfov(camera_width, camera_height, dfov)
    print(fov)
