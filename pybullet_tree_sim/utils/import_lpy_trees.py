#!/usr/bin/env python3
import glob
import os
import shutil
import json

from pybullet_tree_sim import MESHES_PATH


def get_ply_files_in_directory(directory: str) -> list[str]:
    """Get all .ply files in a given directory.

    :param directory: Directory path
    :type directory: str
    :return: List of .ply file paths
    :rtype: list[str]
    """
    ply_files = glob.glob(os.path.join(directory, "**/*.ply"), recursive=True)
    return ply_files


def get_metadata_files_in_directory(directory: str) -> list[str]:
    """Get all metadata JSON files in a given directory.

    :param directory: Directory path
    :type directory: str
    :return: List of metadata JSON file paths
    :rtype: list[str]
    """
    metadata_files = glob.glob(os.path.join(directory, "**/*_metadata.json"), recursive=True)
    return metadata_files


def get_tree_hierarchy_files_in_directory(directory: str) -> list[str]:
    """Get all tree hierarchy JSON files in a given directory.

    :param directory: Directory path
    :type directory: str
    :return: List of tree hierarchy JSON file paths
    :rtype: list[str]
    """
    hierarchy_files = glob.glob(os.path.join(directory, "**/*_hierarchy.json"), recursive=True)
    return hierarchy_files


def merge_hierarchy_with_metadata(hierarchy_path: str, metadata_path: str) -> dict:
    """Merge tree hierarchy JSON with metadata JSON.

    :param hierarchy_path: Path to tree hierarchy JSON file
    :type hierarchy_path: str
    :param metadata_path: Path to metadata JSON file
    :type metadata_path: str
    """
    with open(hierarchy_path, "r") as f:
        hierarchy_data = json.load(f)
    with open(metadata_path, "r") as f:
        cylinder_metadata = json.load(f)

    metadata = {
        "cylinder_data": cylinder_metadata,
        "hierarchy": hierarchy_data,
    }

    return metadata


def sanitize_file_name(filename: str) -> str:
    """Sanitize a filename by replacing spaces with underscores and converting to lowercase.

    :param filename: Original filename
    :type filename: str
    :return: Sanitized filename
    :rtype: str
    """
    sanitized = filename.lower()
    return sanitized


def move_ply_to_trees_dir(source_path: str, dest_directory: str) -> str:
    """Move a .ply file to the specified trees directory.

    :param source_path: Source .ply file path
    :type source_path: str
    :param dest_directory: Destination directory path
    :type dest_directory: str
    :return: New path of the moved .ply file
    :rtype: str
    """
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = os.path.basename(source_path)
    filename = sanitize_file_name(filename)
    dest_path = os.path.join(dest_directory, filename)
    shutil.copy(source_path, dest_path)
    return dest_path


def main():
    dataset_dir = os.path.expanduser("~/dev/lpy_treesim/dataset")
    ply_files = get_ply_files_in_directory(dataset_dir)
    metadata_files = get_metadata_files_in_directory(dataset_dir)
    hierarchy_files = get_tree_hierarchy_files_in_directory(dataset_dir)

    output_ply_dir = os.path.join(MESHES_PATH, "trees", "ply")
    output_metadata_dir = os.path.join(MESHES_PATH, "trees", "metadata")

    for hierarchy_file in sorted(hierarchy_files):
        base_name = os.path.basename(hierarchy_file).replace("_hierarchy.json", "")
        tree_type = base_name.split("_")[1].lower()
        corresponding_metadata_file = os.path.join(dataset_dir, tree_type, base_name + "_metadata.json")
        if os.path.exists(corresponding_metadata_file):
            merged_metadata = merge_hierarchy_with_metadata(hierarchy_file, corresponding_metadata_file)
            if not os.path.exists(output_metadata_dir):
                os.makedirs(output_metadata_dir)
            output_metadata_path = os.path.join(output_metadata_dir, sanitize_file_name(base_name + "_metadata.json"))
            with open(output_metadata_path, "w") as f:
                json.dump(merged_metadata, f, indent=4)

    for ply_file in ply_files:
        move_ply_to_trees_dir(ply_file, output_ply_dir)

    return


if __name__ == "__main__":
    main()
