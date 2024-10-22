#!/usr/bin/env python3
import os
import yaml
import numpy as np

def construct_angle_radians(loader, node):
    """Utility function to construct radian values from yaml."""
    value = loader.construct_scalar(node)
    try:
        return float(value)
    except SyntaxError:
        raise Exception("invalid expression: %s" % value)
        
def construct_angle_degrees(loader, node):
    """Utility function for converting degrees into radians from yaml."""
    return np.radians(construct_angle_radians(loader, node))

def load_yaml(file_path):
    try:
        yaml.SafeLoader.add_constructor("!radians", construct_angle_radians)
        yaml.SafeLoader.add_constructor("!degrees", construct_angle_degrees)
    except Exception:
        raise Exception("yaml support not available; install python-yaml")

    try:
        with open(file_path) as file:
            return yaml.safe_load(file)
    except OSError:  # parent of IOError, OSError *and* WindowsError where available
        return None