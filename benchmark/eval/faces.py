"""Mesh-face resolver for surface-sampled metrics. Human faces are a function of topology
(vertex count); object faces are stored per-result (`obj_faces`) or fall back to the GT object
template when the prediction reuses it (same vertex count). Cached face sets live in eval/assets/.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "assets")
_CACHE = {}


def _load(name):
    if name not in _CACHE:
        _CACHE[name] = np.load(os.path.join(_ASSETS, name)).astype(np.int64)
    return _CACHE[name]


def human_faces(n_verts):
    """SMPL/SMPL-H (6890) or SMPL-X (10475) body faces by vertex count; None if unknown topology."""
    if n_verts == 10475:
        return _load("smplx_faces.npy")
    if n_verts == 6890:
        return _load("smpl_faces.npy")
    return None


def obj_faces(pred_obj_npz, n_verts, gt_obj_faces, gt_obj_nverts):
    """Object faces for a prediction: prefer faces saved in the result npz; else reuse the GT object
    template faces when the method posed that same template (matching vertex count); else None
    (→ vertex resampling for that object, recorded by the caller)."""
    if pred_obj_npz is not None and "obj_faces" in pred_obj_npz:
        f = np.asarray(pred_obj_npz["obj_faces"])
        if f.ndim == 2 and f.shape[1] == 3:
            return f.astype(np.int64)
    if gt_obj_faces is not None and n_verts == gt_obj_nverts:
        return np.asarray(gt_obj_faces).astype(np.int64)
    return None
