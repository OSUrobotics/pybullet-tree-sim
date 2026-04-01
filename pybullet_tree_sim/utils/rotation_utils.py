#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation


def ori_vec_to_quat(ori_vecs: np.ndarray, base_vec: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Convert orientation vectors to quaternions

    :param ori_vecs: Orientation vectors
    :type ori_vecs: np.ndarray
    :param base_vec: Base vector to align from, defaults to np.array([0, 0, 1])
    :type base_vec: np.ndarray, optional
    :return: Quaternions
    :rtype: np.ndarray
    """
    norms = np.linalg.norm(ori_vecs, axis=1, keepdims=True)
    ori_vecs = ori_vecs / norms
    angles = np.arccos(np.dot(ori_vecs, base_vec) / (np.linalg.norm(ori_vecs, axis=1) * np.linalg.norm(base_vec)))
    rot_vecs = np.cross(base_vec, ori_vecs)
    rot_vec_norms = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    rot_vec_norms = np.divide(rot_vecs, rot_vec_norms, where=(rot_vec_norms > 1e-10), out=np.zeros_like(rot_vecs))
    rot_vecs = rot_vec_norms * angles[:, np.newaxis]
    quats = Rotation.from_rotvec(rot_vecs).as_quat()
    return quats


def quat_to_ori_vec(quats: np.ndarray, base_vec: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Convert quaternions to orientation vectors

    :param quats: Quaternions
    :type quats: np.ndarray
    :param base_vec: Base vector to align from, defaults to np.array([0, 0, 1])
    :type base_vec: np.ndarray, optional
    :return: Orientation vectors
    :rtype: np.ndarray
    """
    rot_vecs = Rotation.from_quat(quats)
    ori_vecs = rot_vecs.apply(base_vec)
    return ori_vecs


def quat_to_rot_mat(quats: np.ndarray) -> np.ndarray:
    """Convert quaternions to rotation matrices

    :param quats: Quaternions
    :type quats: np.ndarray
    :return: Rotation matrices
    :rtype: np.ndarray
    """
    rot_mats = Rotation.from_quat(quats).as_matrix()
    return rot_mats
