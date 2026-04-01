#!/usr/bin/env python3
import glob
import os
import shutil
import json
import argparse

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


def sanitize_file_name(filename: str) -> str:
    """Sanitize a filename by replacing spaces with underscores and converting to lowercase.

    :param filename: Original filename
    :type filename: str
    :return: Sanitized filename
    :rtype: str
    """
    sanitized = filename.lower()
    return sanitized


def copy_to_dir(source_path: str, dest_directory: str, overwrite: bool = False) -> str:
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

    if os.path.exists(dest_path) and not overwrite:
        raise FileExistsError("File already exists.")
    shutil.copy(source_path, dest_path)
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Import LPy tree .ply and metadata files into pybullet_tree_sim.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    dataset_dir = os.path.expanduser("~/dev/lpy_treesim/dataset")
    ply_files = get_ply_files_in_directory(dataset_dir)
    metadata_files = get_metadata_files_in_directory(dataset_dir)

    output_ply_dir = os.path.join(MESHES_PATH, "trees", "ply")
    if not os.path.exists(output_ply_dir):
        os.makedirs(output_ply_dir)
    output_metadata_dir = os.path.join(MESHES_PATH, "trees", "metadata")
    if not os.path.exists(output_metadata_dir):
        os.makedirs(output_metadata_dir)

    for m_file in sorted(metadata_files):
        dest_path = copy_to_dir(source_path=m_file, dest_directory=output_metadata_dir, overwrite=args.overwrite)

    for ply_file in ply_files:
        dest_path = copy_to_dir(source_path=ply_file, dest_directory=output_ply_dir, overwrite=args.overwrite)

    return


if __name__ == "__main__":
    main()
