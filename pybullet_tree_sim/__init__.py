import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))


# Global URDF path pointing to robot and supports URDFs
PKL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pkl"))
MESHES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "meshes"))
URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "urdf"))
TEXTURES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "textures")
)
ROBOT_URDF_PATH = os.path.join(
    URDF_PATH, "ur5e", "ur5e_cutter_new_calibrated_precise_level.urdf"
)
CONFIG_PATH = os.path.join(os.path.join(os.path.dirname(__file__), "config"))
CAMERAS_PATH = os.path.join(CONFIG_PATH, "cameras")
TOFS_PATH = os.path.join(CONFIG_PATH, "tofs")
