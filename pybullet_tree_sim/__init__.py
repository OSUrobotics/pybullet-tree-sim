import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))


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
    (30.0/255.0, 60.0/255.0, 10.0/255): "SPUR",
    (51.0/255, 51.0/255, 51.0/255): "TRUNK",
    (153.0/255, 153.0/255, 153.0/255): "BRANCH",
   # (60.0/255, 0.000000, 0.000000): "WATER_BRANCH",
    #(204.0/255.0, 102.0/255.0, 51.0/255.0): "APPLE",
    (36.0/255.0, 117.0/255.0, 32.0/255.0): "LEAF",
}
