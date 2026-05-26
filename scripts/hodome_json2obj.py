"""Export per-frame human + object meshes as OBJ files (for Blender, etc.).

Replaces the legacy SMPL-H per-frame JSON loader with SMPL-X / MHR pose source.
Object rotation/translation still read from mocap/{seq}/object/*.json.

Outputs written to:
  mocap/{seq}/obj/human/{frame:06d}.obj
  mocap/{seq}/obj/object/{frame:06d}.obj
"""
import sys
sys.path.append('./')
import os
from os.path import join
import json
import argparse
import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from utils.rotation_utils import rot6d_to_matrix
from scripts.utils.pose_source import create_pose_source, add_pose_source_args


def setup_device():
    if torch.cuda.is_available():
        d = torch.device("cuda:0"); torch.cuda.set_device(d); return d
    return torch.device("cpu")


def parse_arguments():
    p = argparse.ArgumentParser(description='Export HoDome SMPL-X/MHR + object meshes to OBJ.')
    p.add_argument('--root_path', type=str, default="./HODome")
    p.add_argument('--seq_name', type=str, default="subject02_desk")
    p.add_argument('--start_frame', type=int, default=-1,
                   help='Start frame index. -1 = use startframe.json rgb offset.')
    p.add_argument('--end_frame', type=int, default=-1,
                   help='End frame index (exclusive). -1 = video length.')
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--object_required', action='store_true',
                   help='Skip frames whose object/*.json is missing (default: skip silently)')
    add_pose_source_args(p)
    return p.parse_args()


def load_object_template(root_path, seq_name):
    object_name = seq_name.split('_')[-1]
    p = join(root_path, 'scaned_object', object_name, f'{object_name}_face1000.obj')
    if not os.path.exists(p):
        return None, None
    m = trimesh.load(p, process=False)
    return m, m.vertices.copy()


def load_start_frame(root_path, seq_name):
    sfp = join(root_path, 'startframe.json')
    if not os.path.exists(sfp):
        return 0
    with open(sfp, 'rb') as f:
        d = json.load(f)
    return d['rgb'].get(seq_name, 0)


def main():
    device = setup_device()
    args = parse_arguments()

    mocap_path = join(args.root_path, 'mocap', args.seq_name)
    save_human = join(mocap_path, 'obj', 'human')
    save_object = join(mocap_path, 'obj', 'object')
    os.makedirs(save_human, exist_ok=True)
    os.makedirs(save_object, exist_ok=True)

    print(f"[json2obj] pose source = {args.source}")
    pose = create_pose_source(
        args.source, args.seq_name,
        smplx_npz_dir=args.smplx_npz_dir,
        smplx_model_path=args.smplx_model_path,
        mhr_verts_dir=args.mhr_verts_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    human_faces = pose.faces

    object_mesh, object_mesh_vertices = load_object_template(args.root_path, args.seq_name)
    if object_mesh is None:
        print(f"[json2obj] WARNING: object template missing; skipping object OBJ export")

    start_frame = args.start_frame if args.start_frame >= 0 else load_start_frame(args.root_path, args.seq_name)
    # Determine end frame from video length
    videos_path = join(args.root_path, 'videos', args.seq_name)
    first_vid = sorted(os.listdir(videos_path))[0]
    cap = cv2.VideoCapture(join(videos_path, first_vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    end_frame = args.end_frame if args.end_frame > 0 else total

    for fid in tqdm(range(start_frame, end_frame, args.step), desc='frames'):
        # Human mesh
        try:
            verts = pose.get_vertices(fid)
        except IndexError:
            continue  # pose source has no entry for this frame
        trimesh.Trimesh(vertices=verts, faces=human_faces, process=False).export(
            join(save_human, f'{fid:06d}.obj'))

        # Object mesh
        if object_mesh is not None:
            obj_p = join(mocap_path, 'object', f'{fid:06d}.json')
            if not os.path.exists(obj_p):
                if args.object_required:
                    raise FileNotFoundError(obj_p)
                continue
            with open(obj_p, 'rb') as f:
                obj = json.load(f)
            R6 = np.array(obj['object_R'])
            Rm = rot6d_to_matrix(torch.from_numpy(R6)).numpy().reshape(3, 3).T
            Tv = np.array(obj['object_T']).reshape(1, 3)
            object_mesh.vertices = object_mesh_vertices.dot(Rm.T) + Tv
            object_mesh.export(join(save_object, f'{fid:06d}.obj'), include_texture=False)


if __name__ == "__main__":
    main()
