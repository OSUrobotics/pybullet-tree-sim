import numpy as np

# origins: (N,3), dirs_world: (N,3), mesh: trimesh.Trimesh
locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=origins, ray_directions=dirs_world)

# Prepare output colors (RGBA uint8) default background:
N = len(origins)
colors = np.zeros((N, 4), dtype=np.uint8)  # background 0,0,0,0 or set background color

if len(index_tri) > 0:
    # If mesh has per-face colors
    face_colors = getattr(mesh.visual, "face_colors", None)
    if face_colors is not None and len(face_colors) == len(mesh.faces):
        colors[index_ray] = face_colors[index_tri]
    else:
        # Fall back to per-vertex interpolation
        # triangles: (M,3,3)
        tris = mesh.triangles[index_tri]                         # (M,3,3)
        pts = locations                                         # (M,3)
        v0 = tris[:, 1] - tris[:, 0]
        v1 = tris[:, 2] - tris[:, 0]
        v2 = pts - tris[:, 0]

        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)

        denom = d00 * d11 - d01 * d01
        # avoid division by zero
        valid = denom != 0
        u = np.zeros_like(d00); v = np.zeros_like(d00); w = np.zeros_like(d00)
        v[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
        w[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
        u = 1.0 - v - w

        # gather vertex colors for these triangles: (M,3,4) expected RGBA uint8
        vc = mesh.visual.vertex_colors
        tri_vids = mesh.faces[index_tri]                         # (M,3)
        tri_vcols = vc[tri_vids]                                 # (M,3,4)
        # interpolate
        interp = (u[:, None] * tri_vcols[:, 0, :] +
                  v[:, None] * tri_vcols[:, 1, :] +
                  w[:, None] * tri_vcols[:, 2, :])
        colors[index_ray] = np.clip(interp, 0, 255).astype(np.uint8)