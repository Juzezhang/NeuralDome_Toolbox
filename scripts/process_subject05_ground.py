"""
Process subject05 mocap data to ground-aligned coordinates (mocap_ground).

Recovers the per-date offset from calibration vs calibration_ground,
then applies it to all subject05 sequences to produce NPZ files in mocap_ground/.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Fix numpy compatibility for chumpy (used by MANO pickle files)
import numpy as np
_compat = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
           'object': object, 'str': str, 'unicode': str}
for attr, fallback in _compat.items():
    if not hasattr(np, attr):
        setattr(np, attr, fallback)

import json
import argparse
import numpy as np
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

    # R_ground = R_orig @ R_offset^T  =>  R_offset = (R_orig^T @ R_ground)^T
    R_offset = (R_orig.T @ R_ground).T
    # t_offset = R_ground^T @ (T_orig - T_ground)
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
                     body_model, start_frame):
    """Process a single sequence: transform mocap data and save as NPZ."""
    mocap_path = join(root_path, 'mocap', seq_name)
    smplh_dir = join(mocap_path, 'smplh')
    object_dir = join(mocap_path, 'object')

    total_frames = count_frames(smplh_dir)
    R_offset_tensor = torch.tensor(R_offset, dtype=torch.float64, device=device)

    all_smpl_params = []
    all_object_params = []

    for frame_count in tqdm(range(start_frame, total_frames), desc=f"  {seq_name}"):
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
        object_R = rot6d_to_matrix(object_R).numpy().reshape(3, 3).T
        object_T = np.array(object_data['object_T']).reshape(1, 3)

        # Apply transformation
        object_R = R_offset @ object_R
        object_T = (R_offset @ object_T.T).T
        object_T[0][1] -= avg_offset_y

        all_object_params.append({
            'object_R': object_R,
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
    parser = argparse.ArgumentParser(description='Process subject05 mocap to ground coordinates')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path of hodome dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output directory for NPZ files (mocap_ground)')
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
    with open(join(args.root_path, 'startframe.json')) as f:
        start_frames = json.load(f)

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

    # Find all subject05 sequences with mocap data
    subject05_seqs = []
    for seq in sorted(set(s for d in dataset_info for s in dataset_info[d])):
        if not seq.startswith('subject05'):
            continue
        mocap_dir = join(args.root_path, 'mocap', seq, 'smplh')
        if not os.path.isdir(mocap_dir):
            continue
        date = get_calibration_date(seq, dataset_info)
        if date is None or date not in offsets:
            print(f"  Skipping {seq}: no calibration date or offset found")
            continue
        subject05_seqs.append((seq, date))

    print(f"\nProcessing {len(subject05_seqs)} subject05 sequences...")
    os.makedirs(args.output_path, exist_ok=True)

    for seq, date in subject05_seqs:
        R_offset, avg_offset_y = offsets[date]
        sf = start_frames['rgb'].get(seq, 0)
        process_sequence(seq, args.root_path, args.output_path,
                         R_offset, avg_offset_y, body_model, sf)

    print(f"\nDone! Processed {len(subject05_seqs)} sequences.")


if __name__ == '__main__':
    main()
