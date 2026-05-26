"""Surface point sampling for the BEHAVE-protocol Chamfer (faithful to CHORE/CONTHO).

The papers sample N points on the MESH SURFACE (area-weighted) before Chamfer — CHORE/VisTracker/
StackFLOW/WildHOI use N=10000, CONTHO uses N=6000. This module reproduces that with trimesh's
area-weighted `sample_surface`. When a mesh's faces are unavailable (e.g. HDM's raw point cloud, or
a method whose object template faces we could not recover) we fall back to deterministic vertex
resampling and the caller records that the metric is vertex-based for that mesh.

Determinism: a fixed RNG seed per call so GT-as-pred sanity stays ~0 (same mesh → ~identical samples;
residual is sub-mm sampling noise, expected for a surface-sampled metric).
"""
import numpy as np
import trimesh

N_DEFAULT = 10000  # CHORE/VisTracker/StackFLOW/WildHOI standard (CONTHO=6000); 10000 = field majority


def sample_surface(verts, faces, n=N_DEFAULT, seed=0):
    """Area-weighted surface sampling. faces=None → deterministic vertex resampling to n points."""
    V = np.asarray(verts, dtype=np.float64)
    if faces is None:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(V), size=n) if len(V) < n else rng.permutation(len(V))[:n]
        return V[idx]
    F = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)
