import os

# Global URDF path pointing to robot and supports URDFs
PKL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "pkl"))
MESHES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "meshes"))
URDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "urdf"))
TEXTURES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "textures"))
ROBOT_URDF_PATH = os.path.join(URDF_PATH, "ur5e", "ur5e_cutter_new_calibrated_precise_level.urdf")
CONFIG_PATH = os.path.join(os.path.join(os.path.dirname(__file__), "config"))
CAMERAS_PATH = os.path.join(CONFIG_PATH, "cameras")
TOFS_PATH = os.path.join(CONFIG_PATH, "tofs")
# SUPPORT_AND_POST_PATH = os.path.join(MESHES_PATH, 'urdf', 'supports_and_post.urdf')


RGB_LABEL = {  # RGB colors
    (0.117647, 0.235294, 0.039216): "SPUR",
    (0.313725, 0.313725, 0.313725): "TRUNK",
    (0.254902, 0.176471, 0.058824): "BRANCH",
    (0.235294, 0.000000, 0.000000): "WATER_BRANCH",
}
