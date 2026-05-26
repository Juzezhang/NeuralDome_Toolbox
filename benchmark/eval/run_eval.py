"""Run the HoDome mono-HOI benchmark eval for one method.

GT:   data/gt/{seq}.npz (world frame, SMPL-X) — transformed to view26 CAMERA frame here.
Pred: results/{method}/{seq}.npz, CAMERA frame (view26), meters, with:
        frames (F,), human_joints (F,>=22,3), [human_verts (F,Vh,3)], obj_verts (F,Vo,3)
Metrics (per NeuralDome Table 3): MPJPE, PA-MPJPE (human, 22 body joints), Chamfer (object),
V2V, p.V2V (scene combined-cloud Chamfer). Reports mm. --sanity uses GT-as-pred (expect ~0).
"""
import os, csv, json, argparse, glob
import numpy as np
import metrics as M
import behave_chamfer as BC  # CONTHO/CHORE verbatim Chamfer + scaled-rigid (fair judge metric)
import faces as FA           # mesh-face resolver for surface-sampled metrics
from surface import sample_surface

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = 22  # SMPL-X body joints shared with SMPL


def gt_camera(z):
    """Return dict of GT in view26 camera frame (+ faces for surface sampling)."""
    R, T = z["cam_R"], z["cam_T"]
    jw = z["smplx_joints"][:, :NB, :]
    posed = np.einsum("vd,frd->fvr", z["obj_verts0"], z["obj_R"]) + z["obj_T"][:, None, :]
    return dict(frames=z["frames"],
                joints=M.to_camera(jw, R, T),
                verts=M.to_camera(z["smplx_verts"], R, T),
                obj=M.to_camera(posed, R, T),
                hf=FA.human_faces(z["smplx_verts"].shape[1]),  # SMPL-X faces
                of=np.asarray(z["obj_faces"]) if "obj_faces" in z else None,
                of_nv=int(z["obj_verts0"].shape[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--gt_dir", default=os.path.join(ROOT, "data", "gt"))
    ap.add_argument("--pred_dir", default=None, help="default results/{method}")
    ap.add_argument("--sanity", action="store_true", help="use GT-as-pred (expect ~0)")
    args = ap.parse_args()
    pred_dir = args.pred_dir or os.path.join(ROOT, "results", args.method)
    out_dir = os.path.join(ROOT, "results", args.method); os.makedirs(out_dir, exist_ok=True)

    rows = []
    cov = {"hum_surf": False, "obj_surf": False, "hum_vert": False, "obj_vert": False}
    for gtf in sorted(glob.glob(os.path.join(args.gt_dir, "*.npz"))):
        seq = os.path.basename(gtf)[:-4]
        g = gt_camera(np.load(gtf))
        if args.sanity:
            p = {"frames": g["frames"], "human_joints": g["joints"],
                 "human_verts": g["verts"], "obj_verts": g["obj"]}
            common = list(range(len(g["frames"])))
            pf = g["frames"]
        else:
            pf_path = os.path.join(pred_dir, f"{seq}.npz")
            if not os.path.exists(pf_path):
                continue
            p = dict(np.load(pf_path, allow_pickle=True))
            pf = p["frames"]
        gidx = {f: i for i, f in enumerate(g["frames"])}
        per = {k: [] for k in ("MPJPE", "PA-MPJPE", "Chamfer", "V2V", "p.V2V", "HumanCD", "ObjCD")}
        # faces are per-method constant → resolve once from the result npz (or GT template)
        pred_of = FA.obj_faces(p, np.asarray(p["obj_verts"]).shape[1], g["of"], g["of_nv"])
        pred_hf = (FA.human_faces(np.asarray(p["human_verts"]).shape[1])
                   if "human_verts" in p else None)
        nbad = 0; n_osamp = 0; n_hsamp = 0
        for j, f in enumerate(pf):
            if f not in gidx:
                continue
            gi = gidx[f]
            pj, gj = np.asarray(p["human_joints"])[j][:NB], g["joints"][gi]
            po, go = np.asarray(p["obj_verts"])[j], g["obj"][gi]
            ph = np.asarray(p["human_verts"])[j] if "human_verts" in p else pj
            gh = g["verts"][gi]
            if not (np.isfinite(pj).all() and np.isfinite(po).all() and np.isfinite(ph).all()):
                nbad += 1  # diverged/NaN prediction → skip frame
                continue
            sim = M.umeyama(pj, gj)  # similarity from corresponded body joints
            per["MPJPE"].append(M.mpjpe(pj, gj))
            per["PA-MPJPE"].append(M.pa_mpjpe(pj, gj))
            # object Chamfer (mm): surface-sampled (area-weighted) like the papers, faces-aware
            per["Chamfer"].append(M.chamfer(sample_surface(po, pred_of), sample_surface(go, g["of"])))
            per["V2V"].append(M.v2v(ph, po, gh, go, align=None))
            per["p.V2V"].append(M.v2v(ph, po, gh, go, align=sim))
            # BEHAVE-protocol Chamfer (cm) via CONTHO/CHORE's verbatim code (behave_chamfer.py):
            # scaled-rigid align from 22 corresponded joints, then surface-sample, then their chamfer.
            hcd, ocd, hs, os_ = BC.behave_cd_cm(ph, po, gh, go, pj, gj,
                                                pred_hf=pred_hf, pred_of=pred_of,
                                                gt_hf=g["hf"], gt_of=g["of"])
            per["HumanCD"].append(hcd); per["ObjCD"].append(ocd)
            n_hsamp += hs; n_osamp += os_
        if per["MPJPE"]:
            cov["hum_surf"] = cov["hum_surf"] or n_hsamp > 0
            cov["obj_surf"] = cov["obj_surf"] or n_osamp > 0
            cov["hum_vert"] = cov["hum_vert"] or (len(per["MPJPE"]) - n_hsamp) > 0
            cov["obj_vert"] = cov["obj_vert"] or (len(per["MPJPE"]) - n_osamp) > 0
            rows.append({"seq": seq, "n": len(per["MPJPE"]),
                         **{k: float(np.mean(v)) for k, v in per.items()}})
            print(f"  {seq}: n={len(per['MPJPE'])} MPJPE={rows[-1]['MPJPE']:.1f} "
                  f"PA={rows[-1]['PA-MPJPE']:.1f} Cham={rows[-1]['Chamfer']:.1f} "
                  f"V2V={rows[-1]['V2V']:.1f} pV2V={rows[-1]['p.V2V']:.1f}")

    if not rows:
        print("no seqs evaluated (need predictions in", pred_dir, ")"); return
    keys = ["MPJPE", "PA-MPJPE", "Chamfer", "V2V", "p.V2V", "HumanCD", "ObjCD"]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tot = sum(r["n"] for r in rows)
    with open(os.path.join(out_dir, "metrics_per_seq.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seq", "n"] + keys); w.writeheader(); w.writerows(rows)
    sampling = {"human": "surface" if cov["hum_surf"] and not cov["hum_vert"] else
                ("vertex" if cov["hum_vert"] and not cov["hum_surf"] else "mixed"),
                "object": "surface" if cov["obj_surf"] and not cov["obj_vert"] else
                ("vertex" if cov["obj_vert"] and not cov["obj_surf"] else "mixed"),
                "n_points": 10000}
    json.dump({"method": args.method, "n_seqs": len(rows), "n_frames": tot,
               "metrics_mm": {k: agg[k] for k in ("MPJPE", "PA-MPJPE", "Chamfer", "V2V", "p.V2V")},
               "behave_cd_cm": {"HumanCD": agg["HumanCD"], "ObjCD": agg["ObjCD"]},
               "sampling": sampling},
              open(os.path.join(out_dir, "metrics_summary.json"), "w"), indent=2)
    print(f"  sampling: human={sampling['human']} object={sampling['object']} (N=10000 surface pts)")
    print(f"\n[{args.method}] {len(rows)} seqs, {tot} frames")
    print("  mm: " + "  ".join(f"{k}={agg[k]:.1f}" for k in ("MPJPE", "PA-MPJPE", "Chamfer", "V2V", "p.V2V")))
    print(f"  BEHAVE-CD(cm): HumanCD={agg['HumanCD']:.2f}  ObjCD={agg['ObjCD']:.2f}")


if __name__ == "__main__":
    main()
