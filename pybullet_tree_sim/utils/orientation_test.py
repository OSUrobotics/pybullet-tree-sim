#! /usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import numpy as np

start_z = np.array([0, 0, 1])
start_x = np.array([1, 0, 0])


d = {
    "centroid": [0.0038124999999999973, 0.0006125000000000019, 0.009237499999999997],
    "radius": 0.04124754218497412,
    "length": 0.19351119874642203,
    "orientation": [0.22185955560552698, 0.9626938265960188, -0.15491589273032436],
}

rot_axis = np.cross(start_z, d["orientation"])
rot_axis_norm = np.linalg.norm(rot_axis)

theta = np.arccos(np.dot(start_z, d["orientation"]) / (np.linalg.norm(start_z) * np.linalg.norm(d["orientation"])))

r = R.from_rotvec(rot_axis * theta)

print(r.as_matrix())
