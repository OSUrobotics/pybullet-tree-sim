#!/usr/bin/env python3
import plotly.graph_objects as go
import modern_robotics as mr
from zenlog import log
import numpy as np
import pprint as pp


def debug_sensor_world_data(data):
    pp.pprint(data)
    fig = go.Figure()
    for tof_name, tof_data in data.items():
        # log.warn(tof_data)
        fig.add_trace(
            go.Scatter3d(
                x=tof_data["data"][:, 0],
                y=tof_data["data"][:, 1],
                z=tof_data["data"][:, 2],
                mode="markers",
                marker=dict(size=2),
                name=tof_name,
            )
        )
        inv_view_matrix = mr.TransInv(tof_data["view_matrix"])
        fig.add_trace(
            go.Scatter3d(
                x=[inv_view_matrix[0, 3]],
                y=[inv_view_matrix[1, 3]],
                z=[inv_view_matrix[2, 3]],
                mode="markers",
                name="camera_origin",
                marker=dict(size=5),
            )
        )

        fig.update_layout(
            title=f"World Data",
            scene=dict(
                aspectmode="cube",
                xaxis=dict(range=[-1.0, 1.0]),
                yaxis=dict(range=[-0.1, 1.0]),
                zaxis=dict(range=[-0.0, 2.1]),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.5, y=-1.5, z=1.5),
                ),
            ),
        )
    fig.show()
    return


def debug_deproject_pixels_to_points(sensor, data, cam_coords, world_coords, view_matrix):

    hovertemplate = "id: %{id}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=np.array(list(range(8)) * 8).reshape((8, 8), order="C").flatten(order="F"),
                y=np.array(list(range(8)) * 8),
                # y=np.array([list(range(8))] * 8).T.flatten(order="F"),
                z=data.flatten(),
                mode="markers",
                ids=[f"{i}" for i in range(sensor.depth_width * sensor.depth_height)],
                hovertemplate=hovertemplate,
            )
        ]
    )
    fig.update_layout(
        title="Pixel Coordinates",
        scene=dict(
            aspectmode="cube",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-1.25, z=1.25),
            ),
        ),
    )
    fig.show()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sensor.depth_film_coords[:, 0],
                y=sensor.depth_film_coords[:, 1],
                z=data.flatten(order="F"),
                mode="markers",
                ids=[f"{i}" for i in range(sensor.depth_width * sensor.depth_height)],
                hovertemplate=hovertemplate,
            )
        ]
    )
    fig.update_layout(
        title="Film Coordinates",
        scene=dict(
            aspectmode="cube",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-1.25, z=1.25),
            ),
        ),
    )
    fig.show()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=cam_coords[:, 0],
                y=cam_coords[:, 1],
                z=cam_coords[:, 2],
                mode="markers",
                ids=[f"{i}" for i in range(sensor.depth_width * sensor.depth_height)],
                hovertemplate=hovertemplate,
            )
        ]
    )  # reverse sign of z to match world coords
    fig.update_layout(
        title="Camera Coordinates",
        scene=dict(
            aspectmode="cube",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.25, y=-1.25, z=1.25),
            ),
        ),
    )
    fig.show()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=world_coords[:, 0],
                y=world_coords[:, 1],
                z=world_coords[:, 2],
                name="tof_data",
                mode="markers",
                marker=dict(size=2),
                ids=np.array([f"{i}" for i in range(sensor.depth_width * sensor.depth_height)]),
                hovertemplate=hovertemplate,
            )
        ]
    )
    inv_view_matrix = mr.TransInv(view_matrix)
    fig.add_trace(
        go.Scatter3d(
            x=[inv_view_matrix[0, 3]],
            y=[inv_view_matrix[1, 3]],
            z=[inv_view_matrix[2, 3]],
            mode="markers",
            name="camera_origin",
            marker=dict(size=5),
        )
    )
    fig.update_layout(
        title="World Coordinates",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[-1.0, 1.0]),
            yaxis=dict(range=[-0.1, 1.0]),
            zaxis=dict(range=[-0.0, 2.1]),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=-1.5, z=1.5),
            ),
        ),
    )
    fig.show()

    # log.warn(f"view_matrix: {view_matrix}")
    # log.warn(f"inv_view_matrix: {inv_view_matrix}")
    return
