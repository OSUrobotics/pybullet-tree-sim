#!/usr/bin/env python3
from __future__ import annotations

from enum import Enum


class SensorType(str, Enum):
    """Top-level sensor categories aligned to folders under config/sensors."""

    CAMERA = "camera"
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    TOF = "tof"
    IMU = "imu"


class Modality(str, Enum):
    """Data modalities a sensor can produce."""

    RGB = "rgb"
    DEPTH = "depth"
    POINTCLOUD = "pointcloud"
    RANGE = "range"  # single range measurement (e.g., ToF)
    IMU_ACCEL = "imu_accel"
    IMU_GYRO = "imu_gyro"


class DataType(str, Enum):
    """Strict data type values allowed in sensor YAMLs."""
    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    POINTCLOUD = "pointcloud"
    RANGE = "range"
    IMU = "imu"
    IMU_ACCEL = "imu_accel"
    IMU_GYRO = "imu_gyro"


def parse_data_type(value: str) -> DataType:
    """Validate and parse a YAML `data_type` value into `DataType`.

    :raises `ValueError` for missing or unsupported values.
    """
    if value is None:
        raise ValueError("Missing required 'data_type' in sensor YAML.")
    try:
        return DataType(str(value).strip().lower())
    except ValueError:
        allowed = ", ".join(d.value for d in DataType)
        raise ValueError(f"Unsupported data_type '{value}'. Allowed: {allowed}")


def data_type_modalities(dt: DataType) -> set[Modality]:
    """Map a strict `DataType` to the set of `Modality` it produces."""
    return {
        DataType.RGB: {Modality.RGB},
        DataType.DEPTH: {Modality.DEPTH},
        DataType.RGBD: {Modality.RGB, Modality.DEPTH},
        DataType.POINTCLOUD: {Modality.POINTCLOUD},
        DataType.RANGE: {Modality.RANGE},
        DataType.IMU: {Modality.IMU_ACCEL, Modality.IMU_GYRO},
        DataType.IMU_ACCEL: {Modality.IMU_ACCEL},
        DataType.IMU_GYRO: {Modality.IMU_GYRO},
    }[dt]


def main():
    """Test the types module."""
    dt = parse_data_type("rgbd")
    modalities = data_type_modalities(dt)
    print(f"DataType: {dt}, Modalities: {modalities}")
    return


if __name__ == "__main__":
    main()
