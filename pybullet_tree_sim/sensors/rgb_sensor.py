#!/usr/bin/env python3
from pybullet_tree_sim.sensors.sensor import Sensor


class RGBSensor(Sensor):
    def __init__(self, *args, **kwargs) -> None:
        """TODO: Fill out like the DepthSensor class"""
        super().__init__(*args, **kwargs)

        pbclient = kwargs.get("pbclient")

        return


def main():
    from pybullet_tree_sim.utils.pyb_utils import PyBUtils
    import pprint as pp

    pbutils = PyBUtils(renders=False)
    sensor = RGBSensor(pbclient=pbutils.pbclient, sensor_name="realsense_d435i", sensor_type="camera")
    pp.pprint(sensor.params)
    # pp.pprint(sensor.depth_film_coords)
    # pp.pprint(sensor.depth_proj_mat)
    return


if __name__ == "__main__":
    main()
