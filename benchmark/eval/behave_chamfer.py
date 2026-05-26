"""BEHAVE-protocol Chamfer distance — copied VERBATIM from the methods' own repos for a fair,
identical judge metric across all methods.

Source (identical in both repos — CONTHO copied it from CHORE):
  - CONTHO_RELEASE/lib/utils/eval_utils.py : chamfer_distance, rigid_transform_3D, rigid_align
  - CHORE/recon/eval/chamfer_distance.py   : chamfer_distance (identical function)
This is the exact function CONTHO/CHORE used to produce their reported BEHAVE numbers
(CONTHO Human-CD 4.99 / Object-CD 8.42 cm; CHORE 5.58 / 10.66 cm; PHOSA 12.17 / 26.62 cm).

Verified differences between the two source repos (so we judge fairly with ONE protocol):
  - chamfer_distance FORMULA: IDENTICAL (bidirectional = mean(min_y→x)+mean(min_x→y), sklearn NN, L2).
  - surface sample count: CHORE sample_num=10000, CONTHO sample_num=6000.
  - alignment: both scaled-rigid (Procrustes w/ scale) on the COMBINED human+object, requiring
    pred & GT to share topology (true on BEHAVE: both SMPL-H + same object template).

Our cross-method ADAPTATION (documented; necessary because methods output different body models
— SMPL / SMPL-H / SMPL-X / point cloud — and GT is SMPL-X, so combined-vertex correspondence does
NOT exist): the scaled-rigid transform is estimated from the 22 CORRESPONDED SMPL body joints
(the only cross-topology correspondence) and applied to the combined cloud. The Chamfer is taken on
N=10000 AREA-WEIGHTED SURFACE SAMPLES (CHORE's count; CONTHO uses 6000) — faithful to the papers'
`trimesh.sample.sample_surface` protocol — using each mesh's faces (human by topology, object from
the result npz or the GT template). A mesh without faces (HDM's raw point cloud, or an object whose
template faces we could not recover) falls back to vertex resampling and is recorded as such. Applied
UNIFORMLY to every method → internally fair AND aligned with the papers' surface-sampled numbers.
"""
import numpy as np
from scipy.spatial import cKDTree
from surface import sample_surface, N_DEFAULT


# ---- verbatim FORMULA from CONTHO eval_utils.py / CHORE recon/eval/chamfer_distance.py ----
# The repos use sklearn NearestNeighbors(kd_tree, L2); we use scipy cKDTree, which returns the
# IDENTICAL exact nearest-neighbor distances (verified diff = 0.0) but ~3.6× faster — matters at
# 30 fps. The bidirectional accumulation formula below is unchanged from the source.
def chamfer_distance(x, y, metric='l2', direction='bi'):
    assert metric == 'l2'
    if direction == 'y_to_x':
        chamfer_dist = np.mean(cKDTree(x).query(y)[0])
    elif direction == 'x_to_y':
        chamfer_dist = np.mean(cKDTree(y).query(x)[0])
    elif direction == 'bi':
        min_y_to_x = cKDTree(x).query(y)[0]
        min_x_to_y = cKDTree(y).query(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)  # bidirectional accumulated
    else:
        raise ValueError("direction must be y_to_x / x_to_y / bi")
    return chamfer_dist


# ---- verbatim from CONTHO eval_utils.py ----
def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]; V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)
    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    """Scaled-rigid align A onto B (CONTHO). Requires corresponded A,B (same N)."""
    c, R, t = rigid_transform_3D(A, B)
    return np.transpose(np.dot(c * R, np.transpose(A))) + t


def behave_cd_cm(pred_h, pred_o, gt_h, gt_o, pred_j, gt_j,
                 pred_hf=None, pred_of=None, gt_hf=None, gt_of=None, n=N_DEFAULT):
    """Fair judge Human-CD & Object-CD in cm. Aligns the scene by the scaled-rigid transform
    estimated from the 22 corresponded body joints (cross-topology), then samples N points on each
    mesh SURFACE (area-weighted, faithful to CHORE/CONTHO) and applies their verbatim bidirectional
    chamfer. Faces=None → vertex resampling for that mesh (recorded by the caller). GT in meters →
    ×100 = cm. Returns (human_cd, obj_cd, human_sampled, obj_sampled)."""
    c, R, t = rigid_transform_3D(np.asarray(pred_j, float), np.asarray(gt_j, float))
    ah = (c * (R @ pred_h.T)).T + t
    ao = (c * (R @ pred_o.T)).T + t
    # surface-sample BOTH pred & GT (same N, deterministic seed) before chamfer
    ph_s, gh_s = sample_surface(ah, pred_hf, n), sample_surface(gt_h, gt_hf, n)
    po_s, go_s = sample_surface(ao, pred_of, n), sample_surface(gt_o, gt_of, n)
    human_cd = chamfer_distance(gh_s, ph_s) * 100.0
    obj_cd = chamfer_distance(go_s, po_s) * 100.0
    human_sampled = pred_hf is not None and gt_hf is not None
    obj_sampled = pred_of is not None and gt_of is not None
    return human_cd, obj_cd, human_sampled, obj_sampled
