"""Uniform interface for loading per-frame human vertices from SMPL-X NPZ or MHR vertices NPZ.

Two canonical sources are supported:
  smplx: ./HODome/smplx/{seq}.npz
         (MHR→SMPL-X conversion; loaded + forwarded at runtime via smplx package)
  mhr:   {mhr_verts_dir}/{seq}.npz pre-computed by tools/forward_mhr_to_npz.py
         (MHR mesh vertices, forwarded once in MHR pixi env and saved)

Both expose:
  ps = PoseSource.create(source, seq, **paths_kwargs)
  ps.frame_ids        # (T,) int32
  ps.faces            # (F, 3) int32
  ps.get_vertices(fid_or_row) → (V, 3) float32 in meters, world frame
  ps.num_frames

The MHR mesh is dense (18439 verts, 36874 faces); SMPL-X has 10475 verts, 20908 faces.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_SMPLX_NPZ_DIR = "./HODome/smplx"
DEFAULT_MHR_VERTS_DIR = "./HODome/mhr_verts"
DEFAULT_SMPLX_MODEL_PATH = "models/model_files/smplx/SMPLX_NEUTRAL.npz"


class _SMPLXPoseSource:
    """Loads SMPL-X NPZ (SMPL-X reconstruction) and forwards vertices on demand."""

    def __init__(self, seq, smplx_npz_dir, smplx_model_path, device="cuda"):
        import torch
        import smplx as smplx_pkg

        npz_path = Path(smplx_npz_dir) / f"{seq}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"SMPL-X NPZ not found: {npz_path}")
        self.npz_path = npz_path
        self.data = np.load(npz_path, allow_pickle=True)
        self.frame_ids = self.data["frame_ids"].astype(np.int32)
        self.num_frames = int(self.frame_ids.shape[0])
        self.device = torch.device(device)
        self._torch = torch

        # Build SMPL-X model. The package expects a directory containing the model file.
        model_dir = str(Path(smplx_model_path).parent.parent)
        model_subdir = Path(smplx_model_path).parent.name  # "smplx"
        if model_subdir != "smplx":
            raise ValueError(
                f"smplx_model_path must live under a 'smplx/' subdir; got {smplx_model_path}"
            )
        self.smplx_model = smplx_pkg.create(
            model_dir, model_type="smplx", gender="neutral",
            use_pca=False, flat_hand_mean=True,
            num_betas=10, num_expression_coeffs=10,
            batch_size=1,
        ).to(self.device)
        self.faces = self.smplx_model.faces.astype(np.int32)

    def _row_for(self, fid_or_row):
        if isinstance(fid_or_row, (int, np.integer)):
            # Treat as frame ID — look up the row index
            matches = np.where(self.frame_ids == int(fid_or_row))[0]
            if matches.size == 0:
                raise IndexError(
                    f"frame_id {fid_or_row} not in SMPL-X NPZ {self.npz_path.name}"
                )
            return int(matches[0])
        raise TypeError(f"fid_or_row must be int frame_id, got {type(fid_or_row)}")

    def _forward_row(self, r):
        """Forward SMPL-X for one row, returning the full SMPLXOutput."""
        torch = self._torch
        with torch.no_grad():
            out = self.smplx_model(
                betas=torch.from_numpy(self.data["betas"][r:r+1]).to(self.device),
                expression=torch.from_numpy(self.data["expression"][r:r+1]).to(self.device),
                body_pose=torch.from_numpy(self.data["body_pose"][r:r+1]).to(self.device),
                global_orient=torch.from_numpy(self.data["global_orient"][r:r+1]).to(self.device),
                transl=torch.from_numpy(self.data["transl"][r:r+1]).to(self.device),
                left_hand_pose=torch.from_numpy(self.data["left_hand_pose"][r:r+1]).to(self.device),
                right_hand_pose=torch.from_numpy(self.data["right_hand_pose"][r:r+1]).to(self.device),
                jaw_pose=torch.from_numpy(self.data["jaw_pose"][r:r+1]).to(self.device),
                leye_pose=torch.from_numpy(self.data["leye_pose"][r:r+1]).to(self.device),
                reye_pose=torch.from_numpy(self.data["reye_pose"][r:r+1]).to(self.device),
                return_verts=True,
            )
        return out

    def get_vertices(self, fid_or_row):
        """Forward SMPL-X for one frame and return vertices (V, 3) numpy float32."""
        r = self._row_for(fid_or_row)
        return self._forward_row(r).vertices[0].cpu().numpy().astype(np.float32)

    def get_joints(self, fid_or_row):
        """Forward SMPL-X for one frame and return the 127 joints (J, 3) numpy
        float32, world frame, meters. Useful joint indices for the ego camera:
        head=15, neck=12, left_eye=23, right_eye=24, pelvis=0."""
        r = self._row_for(fid_or_row)
        return self._forward_row(r).joints[0].cpu().numpy().astype(np.float32)

    def get_vertices_batch(self, fids_or_rows):
        """Batch forward for a list of frame IDs. Returns (N, V, 3)."""
        rows = [self._row_for(f) for f in fids_or_rows]
        # Re-create model with the requested batch_size if needed (smplx fixed batch).
        import smplx as smplx_pkg
        torch = self._torch
        Bk = len(rows)
        model = smplx_pkg.create(
            str(Path(self.smplx_model.npz_path if hasattr(self.smplx_model, 'npz_path') else "").parent.parent or "."),
            model_type="smplx", gender="neutral", use_pca=False, flat_hand_mean=True,
            num_betas=10, num_expression_coeffs=10, batch_size=Bk,
        ) if Bk > 1 else self.smplx_model
        if Bk == 1:
            return self.get_vertices(self.frame_ids[rows[0]])[None]

        def t(key):
            return torch.from_numpy(self.data[key][rows]).to(self.device)
        with torch.no_grad():
            out = model.to(self.device)(
                betas=t("betas"), expression=t("expression"),
                body_pose=t("body_pose"), global_orient=t("global_orient"),
                transl=t("transl"),
                left_hand_pose=t("left_hand_pose"), right_hand_pose=t("right_hand_pose"),
                jaw_pose=t("jaw_pose"), leye_pose=t("leye_pose"), reye_pose=t("reye_pose"),
                return_verts=True,
            )
        return out.vertices.cpu().numpy().astype(np.float32)


class _MHRPoseSource:
    """Loads pre-computed MHR vertices NPZ from tools/forward_mhr_to_npz.py."""

    def __init__(self, seq, mhr_verts_dir):
        npz_path = Path(mhr_verts_dir) / f"{seq}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"MHR vertices NPZ not found: {npz_path}. "
                f"Run scripts/forward_mhr_to_npz.py first to pre-compute it."
            )
        self.npz_path = npz_path
        self.data = np.load(npz_path)
        self.vertices_all = self.data["vertices"]      # (T, V, 3)
        self.frame_ids = self.data["frame_ids"].astype(np.int32)
        self.faces = self.data["faces"].astype(np.int32)
        self.num_frames = int(self.frame_ids.shape[0])
        self._fid_to_row = {int(f): i for i, f in enumerate(self.frame_ids)}

    def get_vertices(self, fid_or_row):
        if isinstance(fid_or_row, (int, np.integer)):
            r = self._fid_to_row.get(int(fid_or_row))
            if r is None:
                raise IndexError(
                    f"frame_id {fid_or_row} not in MHR verts NPZ {self.npz_path.name}"
                )
            return self.vertices_all[r].astype(np.float32)
        raise TypeError(f"fid_or_row must be int frame_id, got {type(fid_or_row)}")

    def get_vertices_batch(self, fids_or_rows):
        rows = [self._fid_to_row[int(f)] for f in fids_or_rows]
        return self.vertices_all[rows].astype(np.float32)


def create_pose_source(
    source: str,
    seq: str,
    smplx_npz_dir: Optional[str] = None,
    smplx_model_path: Optional[str] = None,
    mhr_verts_dir: Optional[str] = None,
    device: str = "cuda",
):
    """Factory. source in {smplx, mhr}."""
    source = source.lower()
    if source == "smplx":
        return _SMPLXPoseSource(
            seq=seq,
            smplx_npz_dir=smplx_npz_dir or DEFAULT_SMPLX_NPZ_DIR,
            smplx_model_path=smplx_model_path or DEFAULT_SMPLX_MODEL_PATH,
            device=device,
        )
    if source == "mhr":
        return _MHRPoseSource(
            seq=seq,
            mhr_verts_dir=mhr_verts_dir or DEFAULT_MHR_VERTS_DIR,
        )
    raise ValueError(f"Unknown source '{source}'; expected 'smplx' or 'mhr'")


def add_pose_source_args(parser):
    """Common CLI flags for selecting the human pose source."""
    parser.add_argument("--source", choices=["smplx", "mhr"], default="smplx",
                        help="Human pose source. SMPL-X = SMPL-X reconstruction (default). "
                             "MHR = pre-computed dense mesh vertices.")
    parser.add_argument("--smplx_npz_dir", default=DEFAULT_SMPLX_NPZ_DIR,
                        help="Directory with {seq}.npz SMPL-X files.")
    parser.add_argument("--smplx_model_path", default=DEFAULT_SMPLX_MODEL_PATH,
                        help="SMPLX_NEUTRAL.npz model file path.")
    parser.add_argument("--mhr_verts_dir", default=DEFAULT_MHR_VERTS_DIR,
                        help="Directory with per-seq MHR vertices NPZ.")
    return parser
