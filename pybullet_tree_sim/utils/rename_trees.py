#!/usr/bin/env python3
import glob
import os
from pybullet_tree_sim import MESHES_PATH


def get_files(base_dir: str) -> list[str]:
    files = glob.glob(os.path.join(base_dir, "*.ply"))
    return files


def rename_tree_file(file: str) -> None:
    dirname = os.path.dirname(file)
    tree_type = os.path.split(dirname)[-1]
    basename = os.path.basename(file)
    print(basename)
    ori_tree_name, extension = os.path.splitext(basename)

    _, tree_id = ori_tree_name.split("_")

    new_tree_name = f"tree_{tree_type}_{tree_id.zfill(5)}"
    new_tree_file = os.path.join(dirname, f"{new_tree_name}{extension}")

    # os.rename(file, new_tree_file)

    return


def rename2_(file: str) -> None:
    dirname = os.path.dirname(file)
    basename = os.path.basename(file)
    new_basename = basename.replace("tree", "LPy")
    new_tree_file = os.path.join(dirname, new_basename)
    print(new_tree_file)
    os.rename(file, new_tree_file)
    return


def rename_tree_files(files: list[str]) -> None:
    for file in files:
        rename2_(file)
    return


def main():
    BASE_DIR = os.path.join(MESHES_PATH, "trees_v2", "envy")
    mesh_files = get_files(base_dir=BASE_DIR)
    rename_tree_files(files=mesh_files)

    BASE_DIR = os.path.join(MESHES_PATH, "trees_v2", "ufo")
    mesh_files = get_files(base_dir=BASE_DIR)
    rename_tree_files(files=mesh_files)
    return


if __name__ == "__main__":
    main()
