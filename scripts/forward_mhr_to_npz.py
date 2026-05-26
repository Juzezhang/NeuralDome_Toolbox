"""Pre-compute MHR mesh vertices per sequence — saved as compact NPZ.

The MHR body model requires the MHR python package which only ships in its
own pixi env. We forward MHR once per seq and dump (vertices, frame_ids, faces)
so downstream viz scripts (which may live in a different env) can `np.load`
without needing the MHR package.

Usage:
  python \\
      scripts/forward_mhr_to_npz.py --seq subject04_book --out_dir ./mhr_verts

Produces  <out_dir>/<seq>.npz  with keys:
  vertices  (T, 18439, 3) float32   in meters, world frame
  frame_ids (T,)          int32
  faces     (36874, 3)    int32
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Imported only here so the rest of the toolbox doesn't depend on MHR.
from mhr.mhr import MHR


def load_seq_mhr_jsons(seq, hodome_root, frame_step=1):
    """Read all per-frame MHR JSONs for one seq; returns sorted (frame_ids, mp, ic, fe)."""
    mhr_dir = Path(hodome_root) / "mhr" / seq / "mhr"
    files = sorted(mhr_dir.glob("*.json"))
    fids, mp, ic, fe = [], [], [], []
    for fp in files:
        fid = int(fp.stem)
        d = json.load(open(fp))
        if isinstance(d, list):
            if not d:
                continue
            d0 = d[0]
        else:
            d0 = d["annots"][0] if "annots" in d else d
        if "model_parameters" not in d0:
            continue
        fids.append(fid)
        mp.append(np.asarray(d0["model_parameters"], dtype=np.float32).reshape(-1))
        ic.append(np.asarray(d0["identity_coeffs"], dtype=np.float32).reshape(-1))
        fe.append(np.asarray(d0["face_expr_coeffs"], dtype=np.float32).reshape(-1))
    fids = np.asarray(fids, dtype=np.int32)
    mp = np.stack(mp, 0)
    ic = np.stack(ic, 0)
    fe = np.stack(fe, 0)
    if frame_step > 1:
        sel = slice(None, None, frame_step)
        fids, mp, ic, fe = fids[sel], mp[sel], ic[sel], fe[sel]
    return fids, mp, ic, fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="HoDome sequence name, e.g. subject04_book")
    ap.add_argument("--out_dir", required=True, help="Output directory; will write <out_dir>/<seq>.npz")
    ap.add_argument("--hodome_root", default="./HODome",
                    help="Root of the HoDome dataset (looks for mhr/{seq}/mhr/*.json under here)")
    ap.add_argument("--frame_step", type=int, default=1,
                    help="Sub-sample every Nth frame. Default 1 = all frames.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lod", type=int, default=1, help="MHR mesh LOD (1 = ~18k verts)")
    ap.add_argument("--mhr_assets_dir", default=os.environ.get("MHR_ASSETS_DIR", ""),
                    help="Path to MHR assets/ folder (lod*.fbx + corrective_blendshapes_*.npz). "
                         "Required when `mhr` is installed from PyPI/git (assets are NOT shipped "
                         "with the wheel — download from https://github.com/facebookresearch/MHR/"
                         "releases/download/v1.0.0/assets.zip). Falls back to the MHR_ASSETS_DIR "
                         "env var, then to mhr's bundled default (which only works for editable "
                         "installs from a local checkout).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mhr_fwd] {args.seq} reading JSONs from {args.hodome_root}/mhr/{args.seq}/mhr/...", flush=True)
    fids, mp, ic, fe = load_seq_mhr_jsons(args.seq, args.hodome_root, args.frame_step)
    T = fids.shape[0]
    print(f"[mhr_fwd] T={T} frames after step={args.frame_step}", flush=True)

    print(f"[mhr_fwd] loading MHR (lod={args.lod})...", flush=True)
    t0 = time.perf_counter()
    if args.mhr_assets_dir:
        from pathlib import Path as _P
        mhr_model = MHR.from_files(folder=_P(args.mhr_assets_dir), lod=args.lod, device=device)
    else:
        mhr_model = MHR.from_files(lod=args.lod, device=device)
    print(f"[mhr_fwd]   MHR loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    verts_all = []
    print(f"[mhr_fwd] forwarding MHR in batches of {args.batch}...", flush=True)
    t0 = time.perf_counter()
    for start in range(0, T, args.batch):
        end = min(start + args.batch, T)
        mp_b = torch.from_numpy(mp[start:end]).to(device)
        ic_b = torch.from_numpy(ic[start:end]).to(device)
        fe_b = torch.from_numpy(fe[start:end]).to(device)
        with torch.no_grad():
            verts, _joints = mhr_model.forward(
                identity_coeffs=ic_b,
                model_parameters=mp_b,
                face_expr_coeffs=fe_b,
            )
        # MHR model_parameters [0:6] already include root rotation+translation;
        # output vertices are in cm in world frame. Convert to meters.
        verts = verts / 100.0
        verts_all.append(verts.cpu().numpy())
        if (start // args.batch) % 4 == 0:
            print(f"  batch {start//args.batch+1}/{(T+args.batch-1)//args.batch} ({end}/{T})", flush=True)

    verts_all = np.concatenate(verts_all, 0).astype(np.float32)
    print(f"[mhr_fwd] forward done in {time.perf_counter()-t0:.1f}s", flush=True)

    faces = mhr_model.character_torch.mesh.faces
    if hasattr(faces, "cpu"):
        faces = faces.cpu().numpy()
    faces = np.asarray(faces, dtype=np.int32)

    out_p = Path(args.out_dir) / f"{args.seq}.npz"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, vertices=verts_all, frame_ids=fids, faces=faces)
    print(f"[mhr_fwd] saved {out_p}  verts={verts_all.shape}  faces={faces.shape}", flush=True)


if __name__ == "__main__":
    main()
