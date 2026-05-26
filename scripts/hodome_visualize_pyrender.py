"""HoDome pyrender visualization — solid-shaded SMPL-X / MHR human (+ object) over video.

Mirrors the HOI-M3 pyrender look (pyrender `Renderer`): nice solid shading, color-coded
meshes. Renders the human (SMPL-X or MHR) and, optionally, the interacting object, projected
onto a chosen camera's video frames.

CALIBRATION — IMPORTANT
-----------------------
The human pose source AND the object pose must share a coordinate frame with the chosen
calibration, or the mesh won't land on the actor:
  --source smplx|mhr  → SMPL-X (`smplx/{seq}.npz`) / MHR are in the GROUND-ALIGNED
                        world frame (the reconstruction's world frame).
  object              → `object/{seq}.npz` (object_R/object_T) is ALSO ground-aligned.
  ⇒ use `--calib ground` (default) = `calibration_ground/{date}/calibration.json`.
Do NOT mix with raw `calibration/` (dome frame) — that's for the per-frame `mocap/{seq}/...`
sources, not for SMPL-X / object.

ENV: run in the **hodome** conda env (setup_env.sh installs pyrender + smplx + torch).
For `--source mhr`, pre-compute vertices first with `scripts/forward_mhr_to_npz.py`
(in the `hodome`/pixi env, which has the `mhr` package).

Usage:
  # SMPL-X human + object, cam 0, full video → MP4
  python scripts/hodome_visualize_pyrender.py --seq_name subject08_trashcan --source smplx \
      --views 0 --output output/subject08_trashcan_smplx.mp4

  # MHR human (needs pre-computed verts), no object
  python scripts/hodome_visualize_pyrender.py --seq_name subject08_trashcan --source mhr \
      --mhr_verts_dir /path/to/mhr_verts --no-object --views 0 --output out.mp4
"""
import sys
sys.path.append('./')
import os
from os.path import join
import json
import argparse
import cv2
import numpy as np
import trimesh
from tqdm import tqdm

from viz.pyrender_wrapper import Renderer
from scripts.utils.pose_source import create_pose_source, add_pose_source_args

# vid → color (palette): 0=blue-ish human, 1=yellow object
HUMAN_VID = 0
OBJECT_VID = 1


def seq_date(root, seq):
    info = json.load(open(join(root, 'dataset_information.json')))
    for d in ('20221018', '20221019', '20221020'):
        if seq in info.get(d, []):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root_path', default='./HODome', help='HoDome dataset root')
    ap.add_argument('--seq_name', required=True)
    ap.add_argument('--views', default='0', help='comma-sep raw cam ids (data{N+1}.mp4), e.g. 0,9,30')
    ap.add_argument('--calib', choices=['ground', 'raw'], default='ground',
                    help="ground=calibration_ground (matches SMPL-X/MHR + object/ pose); "
                         "raw=calibration (dome frame, only for mocap/ per-frame sources)")
    ap.add_argument('--object', dest='object', action=argparse.BooleanOptionalAction, default=True,
                    help='render the interacting object from object/{seq}.npz')
    ap.add_argument('--resolution', type=int, default=720)
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--end_frame', type=int, default=-1)
    ap.add_argument('--frame_step', type=int, default=1)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--output', required=True, help='MP4 path (single view) or dir (multi view)')
    add_pose_source_args(ap)
    args = ap.parse_args()

    print(f"[viz] source={args.source}  calib={args.calib}  object={args.object}")
    pose = create_pose_source(
        args.source, args.seq_name,
        smplx_npz_dir=args.smplx_npz_dir, smplx_model_path=args.smplx_model_path,
        mhr_verts_dir=args.mhr_verts_dir,
        device='cuda',
    )
    human_faces = pose.faces

    # Object (ground-aligned, mean-centered template + per-frame R/T)
    obj_mesh = obj_verts0 = obj_R = obj_T = None
    if args.object:
        oname = args.seq_name.split('_')[-1]
        otmpl = join(args.root_path, 'scaned_object', oname, f'{oname}_face1000.obj')
        opose = join(args.root_path, 'object', f'{args.seq_name}.npz')
        if os.path.exists(otmpl) and os.path.exists(opose):
            obj_mesh = trimesh.load(otmpl, process=False)
            obj_verts0 = obj_mesh.vertices.copy() - obj_mesh.vertices.mean(0)
            od = np.load(opose)
            obj_R, obj_T = np.array(od['object_R']), np.array(od['object_T'])
            print(f"[viz] object '{oname}': {obj_verts0.shape[0]} verts, {obj_R.shape[0]} frames")
        else:
            print(f"[viz] object template/pose missing → skipping object")
            args.object = False

    # Calibration
    date = seq_date(args.root_path, args.seq_name)
    calib_dir = 'calibration_ground' if args.calib == 'ground' else 'calibration'
    cam = json.load(open(join(args.root_path, calib_dir, date, 'calibration.json')))

    renderer = Renderer()
    scaling = 2160 / args.resolution
    views = [int(v) for v in args.views.split(',')]
    multi = len(views) > 1
    if multi:
        os.makedirs(args.output, exist_ok=True)

    for vi in views:
        K = np.array(cam[str(vi)]['K']).reshape(3, 3) / scaling
        K[2, 2] = 1
        RT = np.array(cam[str(vi)]['RT']).reshape(4, 4)
        R, T = RT[:3, :3], RT[:3, 3:].reshape(3)
        vpath = join(args.root_path, 'videos', args.seq_name, f'data{vi + 1}.mp4')
        if not os.path.exists(vpath):
            print(f"[viz] skip view {vi}: no video"); continue
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = args.end_frame if args.end_frame > 0 else min(total, pose.num_frames)
        end = min(end, total, pose.num_frames)
        ow, oh = int(3840 / scaling), int(2160 / scaling)
        outp = args.output if not multi else join(args.output, f'{args.seq_name}_cam{vi:02d}.mp4')
        writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (ow, oh))
        print(f"[viz] view {vi} → {outp}  ({end} frames)")

        for f in tqdm(range(args.start_frame, end, args.frame_step), desc=f'cam{vi}'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            img_rgb = cv2.cvtColor(cv2.resize(img, (ow, oh)), cv2.COLOR_BGR2RGB)  # BGR→RGB for pyrender
            try:
                hv = pose.get_vertices(f)
            except IndexError:
                continue
            render_data = {HUMAN_VID: {'vertices': hv, 'faces': human_faces, 'vid': HUMAN_VID, 'name': 'human'}}
            if args.object and obj_R is not None and f < len(obj_R):
                ov = obj_verts0.dot(obj_R[f].T) + obj_T[f]
                render_data[OBJECT_VID] = {'vertices': ov, 'faces': obj_mesh.faces, 'vid': OBJECT_VID, 'name': 'object'}
            rgb = renderer.render_image(render_data, img_rgb, {'K': K, 'R': R, 'T': T}, [])[2][0]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])  # RGB→BGR for cv2
        cap.release(); writer.release()
    print("[viz] done.")


if __name__ == '__main__':
    main()
