"""Metric primitives for the HoDome mono-HOI benchmark (NeuralDome Table-3 style).

All inputs are numpy arrays in a common coordinate frame, meters.
- MPJPE: mean per-joint position error after ROOT (pelvis) alignment.
- PA-MPJPE: after similarity (Procrustes) alignment.
- Chamfer: symmetric nearest-neighbor distance between two point sets (object).
- V2V / p.V2V: scene (human+object) vertex error; since methods output different human
  topologies than the SMPL-X GT, V2V is computed as symmetric Chamfer over the COMBINED
  human+object point cloud (topology-free), absolute (V2V) and Procrustes-aligned (p.V2V).
Units returned in millimeters.
"""
import numpy as np
from scipy.spatial import cKDTree

M2MM = 1000.0


def umeyama(S, T):
    """Similarity transform (scale, R, t) mapping S onto T (Umeyama). S,T: (N,3) CORRESPONDED."""
    mu_s, mu_t = S.mean(0), T.mean(0)
    Sc, Tc = S - mu_s, T - mu_t
    cov = Tc.T @ Sc / S.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt
    var_s = (Sc ** 2).sum() / S.shape[0]
    scale = np.trace(np.diag(D) @ Dm) / var_s
    t = mu_t - scale * R @ mu_s
    return scale, R, t


def apply_sim(X, sim):
    """Apply (scale, R, t) to X (...,3)."""
    s, R, t = sim
    return (s * (R @ X.T)).T + t


def procrustes_align(S, T):
    """Convenience: align corresponded S onto T (used for joints, |S|==|T|)."""
    return apply_sim(S, umeyama(S, T))


def mpjpe(pred_j, gt_j, root=0):
    """Root-aligned mean per-joint position error (mm). pred_j, gt_j: (J,3)."""
    p = pred_j - pred_j[root]; g = gt_j - gt_j[root]
    return np.linalg.norm(p - g, axis=-1).mean() * M2MM


def pa_mpjpe(pred_j, gt_j):
    """Procrustes-aligned MPJPE (mm)."""
    return np.linalg.norm(procrustes_align(pred_j, gt_j) - gt_j, axis=-1).mean() * M2MM


def chamfer(pred_pts, gt_pts):
    """Symmetric Chamfer distance (mm): mean of both nearest-neighbor directions."""
    da, _ = cKDTree(gt_pts).query(pred_pts)
    db, _ = cKDTree(pred_pts).query(gt_pts)
    return 0.5 * (da.mean() + db.mean()) * M2MM


def v2v(pred_h, pred_o, gt_h, gt_o, align=None):
    """Scene V2V via combined-cloud symmetric Chamfer (mm). *_h human verts, *_o object verts.
    Topology-free: pred and GT clouds may differ in size. For p.V2V pass `align` = the
    (scale,R,t) similarity from the corresponded body joints (umeyama(pred_j, gt_j)); it is
    applied to the combined pred cloud before Chamfer (correspondence-free)."""
    P = np.concatenate([pred_h, pred_o], 0)
    G = np.concatenate([gt_h, gt_o], 0)
    if align is not None:
        P = apply_sim(P, align)
    return chamfer(P, G)


def to_camera(X_world, R, T):
    """World (ground) -> view26 camera frame. X_world (...,3), R (3,3), T (3,)."""
    return X_world @ R.T + T
