"""
DEPRECATED — kept for historical reference only.

Historically generated `mocap_ground_new/{seq}_{human,object}.npz` (SMPL-H, ground-aligned).
That directory has since been split into the top-level `smplh/{seq}.npz` (human) and
`object/{seq}.npz` (object). The newer canonical pose source for HoDome is the SOMA-X
SMPL-X NPZ at
  /simurgh2/users/juze/datasets/hodome/smplx/{seq}.npz
which includes locked-identity early-frame fits and per-frame fitting errors.
New viz scripts (hodome_visualization.py / hodome_visualization_npz.py /
hodome_json2obj.py) default to SMPL-X via scripts/utils/pose_source.py.

If you still need the SMPL-H legacy output, re-run with --force.

----
Process ALL subjects' mocap data to ground-aligned coordinates (mocap_ground_new).

Key differences from process_subject05_ground.py:
  - Processes all subjects (01-10), not just subject05
  - start_frame = 0 (includes all frames)
  - Object pose adjusted for centered mesh: T += mean_V @ R_local.T
  - Output to mocap_ground_new/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Deprecation gate
if __name__ == "__main__" and "--force" not in sys.argv:
    print(
        "[process_all_ground.py] DEPRECATED — mocap_ground_new has been superseded by "
        "the SOMA-X SMPL-X NPZ at\n"
        "  /simurgh2/users/juze/datasets/hodome/smplx/\n"
        "Pass --force to run anyway, or update your script to load SMPL-X via\n"
        "  scripts.utils.pose_source.create_pose_source('smplx', seq_name).",
        file=sys.stderr,
    )
    sys.exit(0)
elif "--force" in sys.argv:
    sys.argv.remove("--force")

# Fix numpy compatibility for chumpy (used by MANO pickle files)
import numpy as np
_compat = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
           'object': object, 'str': str, 'unicode': str}
for attr, fallback in _compat.items():
    if not hasattr(np, attr):
        setattr(np, attr, fallback)

import json
import argparse
import trimesh
import torch
from tqdm import tqdm
from os.path import join
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from models.body_models_easymocap.smplx import SMPLHModel
from models.body_models_easymocap.lbs import lbs, batch_rodrigues
from models.body_models_easymocap.smpl import to_tensor
from utils.rotation_utils import rot6d_to_matrix
from yacs.config import CfgNode

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def recover_offset_from_calibration(cal_orig_path, cal_ground_path):
    """Recover R_offset and avg_offset_y by comparing calibration and calibration_ground."""
    with open(cal_orig_path, 'rb') as f:
        cal_orig = json.load(f)
    with open(cal_ground_path, 'rb') as f:
        cal_ground = json.load(f)

    RT_orig = np.array(cal_orig['0']['RT']).reshape(4, 4)
    RT_ground = np.array(cal_ground['0']['RT']).reshape(4, 4)
    R_orig = RT_orig[:3, :3]
    R_ground = RT_ground[:3, :3]
    T_orig = RT_orig[:3, 3]
    T_ground = RT_ground[:3, 3]

    R_offset = (R_orig.T @ R_ground).T
    t_offset = R_ground.T @ (T_orig - T_ground)
    avg_offset_y = -t_offset[1]

    return R_offset, avg_offset_y


def convert_to_standard_smpl_Th(body_model, params):
    """Convert Th to standard SMPL convention: Tnew = Th - j0 + rot @ j0."""
    params_checked = body_model.check_params(params)
    poses = params_checked['poses']
    shapes = params_checked['shapes']
    Th = params_checked['Th']
    Rh = params_checked['Rh']

    _, joints, _, _ = lbs(
        shapes, poses, body_model.v_template,
        body_model.shapedirs, body_model.posedirs,
        body_model.J_regressor, body_model.parents,
        body_model.weights, pose2rot=True, dtype=body_model.dtype, only_shape=True
    )
    j0 = joints[:, 0, :]
    rot = batch_rodrigues(Rh)
    Tnew = Th - j0 + torch.einsum('bij,bj->bi', rot, j0)

    return dict(
        poses=poses.detach().cpu().numpy(),
        shapes=shapes.detach().cpu().numpy(),
        Rh=Rh.detach().cpu().numpy(),
        Th=Tnew.detach().cpu().numpy(),
    )


def get_calibration_date(seq_name, dataset_info):
    """Determine which calibration date a sequence belongs to."""
    for date in ['20221018', '20221019', '20221020']:
        if seq_name in dataset_info[date]:
            return date
    return None


def count_frames(smplh_dir):
    """Count the number of JSON frame files in a directory."""
    return len([f for f in os.listdir(smplh_dir) if f.endswith('.json')])


def process_sequence(seq_name, root_path, output_path, R_offset, avg_offset_y,
                     body_model, mean_V):
    """Process a single sequence: transform mocap data and save as NPZ.

    Args:
        mean_V: mean vertex position of the object mesh (shape (3,)), used to
                adjust object T for mesh centering.
    """
    mocap_path = join(root_path, 'mocap', seq_name)
    smplh_dir = join(mocap_path, 'smplh')
    object_dir = join(mocap_path, 'object')

    total_frames = count_frames(smplh_dir)
    R_offset_tensor = torch.tensor(R_offset, dtype=torch.float64, device=device)

    all_smpl_params = []
    all_object_params = []

    for frame_count in tqdm(range(0, total_frames), desc=f"  {seq_name}"):
        frame_str = str(frame_count).zfill(6)

        # --- SMPL ---
        smpl_param_path = join(smplh_dir, f'{frame_str}.json')
        with open(smpl_param_path, 'rb') as f:
            smpl_params = json.load(f)['annots'][0]
        smpl_params = {
            key: np.array(value) if isinstance(value, list) else value
            for key, value in smpl_params.items()
        }

        # Transform Rh: R_offset @ Rh_matrix
        Rh_matrix = axis_angle_to_matrix(
            torch.tensor(smpl_params['Rh'], device=device)
        )[0]  # (3,3)
        Rh_new_matrix = R_offset_tensor @ Rh_matrix
        smpl_params['Rh'] = matrix_to_axis_angle(
            Rh_new_matrix.unsqueeze(0)
        ).cpu().numpy()

        # Transform Th: R_offset @ Th, then subtract avg_offset_y
        smpl_params['Th'] = (R_offset @ smpl_params['Th'].T).T
        smpl_params['Th'][0][1] -= avg_offset_y

        # Convert Th to standard SMPL convention
        smpl_params = convert_to_standard_smpl_Th(body_model, smpl_params)

        all_smpl_params.append({
            'shapes': smpl_params['shapes'],
            'poses': smpl_params['poses'],
            'Rh': smpl_params['Rh'],
            'Th': smpl_params['Th'],
        })

        # --- Object ---
        object_path = join(object_dir, f'{frame_str}.json')
        with open(object_path, 'rb') as f:
            object_data = json.load(f)

        object_R = np.array(object_data['object_R'])
        object_R = torch.from_numpy(object_R)
        R_local = rot6d_to_matrix(object_R).numpy().reshape(3, 3).T
        object_T = np.array(object_data['object_T']).reshape(1, 3)

        # Adjust T for mesh centering: T_new = T_old + mean_V @ R_local.T
        object_T += mean_V @ R_local.T

        # Apply ground transformation
        object_R_ground = R_offset @ R_local
        object_T = (R_offset @ object_T.T).T
        object_T[0][1] -= avg_offset_y

        all_object_params.append({
            'object_R': object_R_ground,
            'object_T': object_T,
        })

    # Save NPZ files
    output_human_path = join(output_path, f"{seq_name}_human.npz")
    output_object_path = join(output_path, f"{seq_name}_object.npz")

    np.savez(output_human_path,
             betas=np.array([p['shapes'] for p in all_smpl_params]),
             poses=np.array([p['poses'] for p in all_smpl_params]),
             Rh=np.array([p['Rh'] for p in all_smpl_params]),
             Th=np.array([p['Th'] for p in all_smpl_params]),
             model='smplh',
             gender='neutral',
             mocap_frame_rate=60)

    np.savez(output_object_path,
             object_R=np.array([p['object_R'] for p in all_object_params]),
             object_T=np.array([p['object_T'] for p in all_object_params]),
             mocap_frame_rate=60)

    print(f"  Saved {len(all_smpl_params)} frames -> {output_human_path}")
    print(f"  Saved {len(all_object_params)} frames -> {output_object_path}")


def main():
    parser = argparse.ArgumentParser(description='Process all mocap to ground coordinates with mesh-centered object poses')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path of hodome dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for NPZ files (mocap_ground_new)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to SMPLH model file (default: auto-detect)')
    parser.add_argument('--mano_path', type=str, default=None,
                        help='Path to MANO model directory (default: auto-detect)')
    args = parser.parse_args()

    # Auto-detect model paths
    toolbox_root = os.path.join(os.path.dirname(__file__), '..')
    if args.model_path is None:
        args.model_path = os.path.join(toolbox_root, 'models/model_files/smplhv1.2/neutral/model.npz')
    if args.mano_path is None:
        args.mano_path = os.path.join(toolbox_root, 'models/model_files/manov1.2')

    # Load dataset info
    with open(join(args.root_path, 'dataset_information.json')) as f:
        dataset_info = json.load(f)

    # Initialize body model (for convert_to_standard_smpl_Th)
    cfg_hand = CfgNode()
    cfg_hand.use_pca = True
    cfg_hand.use_flat_mean = False
    cfg_hand.num_pca_comps = 12
    body_model = SMPLHModel(
        model_path=args.model_path,
        mano_path=args.mano_path,
        cfg_hand=cfg_hand,
        NUM_SHAPES=16,
    )

    # Recover offsets from calibration comparison
    print("Recovering offsets from calibration data...")
    offsets = {}
    for date in ['20221018', '20221019', '20221020']:
        cal_orig = join(args.root_path, 'calibration', date, 'calibration.json')
        cal_ground = join(args.root_path, 'calibration_ground', date, 'calibration.json')
        if os.path.exists(cal_orig) and os.path.exists(cal_ground):
            R_offset, avg_offset_y = recover_offset_from_calibration(cal_orig, cal_ground)
            offsets[date] = (R_offset, avg_offset_y)
            print(f"  {date}: avg_offset_y = {avg_offset_y:.6f}")

    # Pre-load and cache object mesh mean vertices
    print("Loading object mesh mean vertices...")
    object_mean_V_cache = {}
    scaned_object_dir = join(args.root_path, 'scaned_object')
    for obj_name in sorted(os.listdir(scaned_object_dir)):
        obj_mesh_path = join(scaned_object_dir, obj_name, f'{obj_name}_face1000.obj')
        if os.path.isfile(obj_mesh_path):
            mesh = trimesh.load(obj_mesh_path, process=False)
            object_mean_V_cache[obj_name] = mesh.vertices.mean(axis=0)
            print(f"  {obj_name}: mean_V = {object_mean_V_cache[obj_name]}")

    # Discover all sequences with mocap data
    mocap_dir = join(args.root_path, 'mocap')
    all_seqs = []
    for seq in sorted(os.listdir(mocap_dir)):
        smplh_path = join(mocap_dir, seq, 'smplh')
        object_path = join(mocap_dir, seq, 'object')
        if not os.path.isdir(smplh_path) or not os.path.isdir(object_path):
            continue
        date = get_calibration_date(seq, dataset_info)
        if date is None or date not in offsets:
            print(f"  Skipping {seq}: no calibration date or offset found")
            continue
        obj_name = seq.split('_')[-1]
        if obj_name not in object_mean_V_cache:
            print(f"  Skipping {seq}: no mesh found for object '{obj_name}'")
            continue
        all_seqs.append((seq, date, obj_name))

    print(f"\nProcessing {len(all_seqs)} sequences...")
    os.makedirs(args.output_path, exist_ok=True)

    for i, (seq, date, obj_name) in enumerate(all_seqs):
        R_offset, avg_offset_y = offsets[date]
        mean_V = object_mean_V_cache[obj_name]
        print(f"\n[{i+1}/{len(all_seqs)}] {seq} (date={date})")
        process_sequence(seq, args.root_path, args.output_path,
                         R_offset, avg_offset_y, body_model, mean_V)

    print(f"\nDone! Processed {len(all_seqs)} sequences.")


if __name__ == '__main__':
    main()
