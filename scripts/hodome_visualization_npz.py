"""Visualize HoDome sequences using SMPL-X (default) or MHR mesh.

Pose source is selected via --source:
  smplx  → SMPL-X NPZ at {smplx_npz_dir}/{seq}.npz (canonical)
  mhr    → pre-computed MHR vertices NPZ from scripts/forward_mhr_to_npz.py

Camera calibration uses calibration_ground/ (ground-aligned dome cameras).
Object pose comes from object/{seq}.npz; pass
--object_source=object (default) or --object_source=none to skip.

Output: per-frame JPGs at {output_path}/{seq}/{view}/{frame}.jpg
"""
import sys
sys.path.append('./')
import os
from os.path import join
import json
import cv2
import numpy as np
import torch
import argparse
import trimesh
from tqdm import tqdm

from viz.pyt3d_wrapper import Pyt3DWrapper
from scripts.utils.pose_source import create_pose_source, add_pose_source_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

parser = argparse.ArgumentParser(description='NeuralDome HoDome viz (SMPL-X / MHR)')
parser.add_argument('--root_path', type=str, default="./HODome")
parser.add_argument('--seq_name', type=str, default="subject02_baseball")
parser.add_argument('--resolution', type=int, default=720)
parser.add_argument('--output_path', type=str, default="output/")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--end_frame', type=int, default=-1)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--views', default='9', help='Comma-sep view indices to render (e.g. 0,9,30) or "all"')
parser.add_argument('--object_source', choices=['object', 'none'],
                    default='object', help='Where to read object R/T from')
add_pose_source_args(parser)
args = parser.parse_args()

# Setup paths
dataset_info_path = join(args.root_path, 'dataset_information.json')
object_path_dir = join(args.root_path, 'object')

# Pose source (replaces SMPL-H BodyModel)
print(f"[viz] pose source = {args.source}")
pose = create_pose_source(
    args.source, args.seq_name,
    smplx_npz_dir=args.smplx_npz_dir,
    smplx_model_path=args.smplx_model_path,
    mhr_verts_dir=args.mhr_verts_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# Object template
object_mesh = None
object_R_all = object_T_all = None
if args.object_source == 'object':
    object_name = args.seq_name.split('_')[-1]
    object_template_path = join(args.root_path, 'scaned_object', object_name, f'{object_name}_face1000.obj')
    if os.path.exists(object_template_path):
        object_mesh = trimesh.load(object_template_path, process=False)
        object_mesh_vertices = object_mesh.vertices.copy()
        object_mesh_vertices -= object_mesh_vertices.mean(0)
        object_path = join(object_path_dir, f"{args.seq_name}.npz")
        if os.path.exists(object_path):
            object_data = np.load(object_path)
            object_R_all = np.array(object_data['object_R'])
            object_T_all = np.array(object_data['object_T'])
        else:
            print(f"[viz] WARNING: {object_path} missing; skipping object render")
            object_mesh = None
    else:
        print(f"[viz] WARNING: {object_template_path} missing; skipping object render")

# Calibration (ground-aligned for *_npz script)
with open(dataset_info_path, 'rb') as file:
    dataset_info = json.load(file)
calibration_dates = ['20221018', '20221019', '20221020']
calibration_path = ''
for date in calibration_dates:
    if args.seq_name in dataset_info[date]:
        calibration_path = join(args.root_path, 'calibration_ground', date, 'calibration.json')
        break
if not calibration_path:
    sys.exit("Calibration data not found for the given sequence name.")
with open(calibration_path, 'rb') as file:
    camera_params = json.load(file)

videos_path = join(args.root_path, 'videos', args.seq_name)
video_files = sorted(os.listdir(videos_path))

# Resolve which views to render
if args.views == 'all':
    view_indices = [int(f.split('.')[0][4:]) - 1 for f in video_files]
else:
    view_indices = [int(v) for v in args.views.split(',')]

for view_index in view_indices:
    video_name = f"data{view_index + 1}.mp4"
    video_path = join(videos_path, video_name)
    if not os.path.exists(video_path):
        print(f"[viz] skip view {view_index}: {video_name} missing")
        continue
    output_dir = join(args.output_path, args.seq_name, str(view_index))
    os.makedirs(output_dir, exist_ok=True)
    print(f'[viz] rendering view {view_index}')

    scaling_factor = 2160 / args.resolution
    K = np.array(camera_params[str(view_index)]['K']).reshape((3, 3)) / scaling_factor
    K[2, 2] = 1
    RT = np.array(camera_params[str(view_index)]['RT']).reshape((4, 4))
    R, T = RT[:3, :3], RT[:3, 3:].reshape(3)

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    npz_frames = pose.num_frames
    end_frame = args.end_frame if args.end_frame > 0 else min(total_frames, npz_frames)
    end_frame = min(end_frame, total_frames, npz_frames)

    human_faces = pose.faces

    for frame_count in tqdm(range(args.start_frame, end_frame, args.step),
                            desc=f"view {view_index}"):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ok, img = capture.read()
        if not ok:
            break
        img = cv2.resize(img[:, :, ::-1], (int(3840 / scaling_factor), int(2160 / scaling_factor)))

        # Human vertices from pose source
        verts = pose.get_vertices(frame_count)
        human_mesh = trimesh.Trimesh(vertices=verts, faces=human_faces, process=False)

        meshes = [human_mesh]
        if object_mesh is not None and object_R_all is not None:
            object_mesh.vertices = (object_mesh_vertices.dot(object_R_all[frame_count].T)
                                    + object_T_all[frame_count])
            meshes.append(object_mesh)

        pyt3d_wrapper = Pyt3DWrapper(image_size=(720, 1280), K=K, R=R, T=T, image=img)
        rendered_image = pyt3d_wrapper.render_meshes(meshes, False, R=R, T=T)
        save_path = join(output_dir, f'{str(frame_count).zfill(6)}.jpg')
        cv2.imwrite(save_path, 255 * rendered_image[:, :, ::-1])
    capture.release()
