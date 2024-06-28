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
from models.body_models_easymocap.smplx import SMPLHModel
from yacs.config import CfgNode
from utils.rotation_utils import rot6d_to_matrix


# Set up the computation device based on CUDA availability
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='NeuralDome command line tools')
    parser.add_argument('--root_path', type=str, default="/nas/nas_10/NeuralDome/Hodome")
    parser.add_argument('--seq_name', type=str, default="subject02_desk")
    return parser.parse_args()


# Load configuration for SMPL model
def load_smpl_model(device):
    cfg_hand = CfgNode()
    cfg_hand.use_pca = True
    cfg_hand.use_flat_mean = False
    cfg_hand.num_pca_comps = 12
    body_model = SMPLHModel(model_path='models/model_files/smplhv1.2/neutral/model.npz',
                            device=device, mano_path='models/model_files/manov1.2',
                            cfg_hand=cfg_hand, NUM_SHAPES=16)
    return body_model


# Load object template
def load_object_template(root_path, seq_name):
    object_name = seq_name.split('_')[-1]
    object_template_path = join(root_path, 'scaned_object', object_name, f'{object_name}_face1000.obj')
    object_mesh = trimesh.load(object_template_path, process=False)
    object_mesh_vertices = object_mesh.vertices
    return object_mesh, object_mesh_vertices


# Load start frame information
def load_start_frame_info(start_frame_path, seq_name):
    with open(start_frame_path, 'rb') as file:
        start_frames = json.load(file)
    return start_frames['rgb'][seq_name]


# Process frames and save human and object meshes
def process_frames(start_frame, total_frames, mocap_path, save_path_human, save_path_object, body_model, object_mesh,
                   object_mesh_vertices):
    for frame_idx in tqdm(range(start_frame, total_frames), desc="Processing frames"):
        smpl_param_path = join(mocap_path, 'smplh', f'{str(frame_idx).zfill(6)}.json')
        with open(smpl_param_path, 'rb') as file:
            smpl_params = json.load(file)['annots'][0]
        smpl_params = {key: np.array(value) if isinstance(value, list) else value for key, value in smpl_params.items()}

        # Generate human mesh from SMPL parameters
        output_mesh = body_model(**smpl_params)
        human_mesh = trimesh.Trimesh(vertices=output_mesh[0].cpu().numpy(), faces=body_model.faces)
        human_mesh_file = join(save_path_human, f'{str(frame_idx).zfill(6)}.obj')
        human_mesh.export(human_mesh_file)

        # Load and apply object transformation
        object_RT_path = join(mocap_path, 'object/refine/json', f'{str(frame_idx).zfill(6)}.json')
        with open(object_RT_path, 'rb') as file:
            object_transformation = json.load(file)
        object_rotation = np.array(object_transformation['object_R'])
        object_rotation = torch.from_numpy(object_rotation)
        object_rotation = rot6d_to_matrix(object_rotation).numpy().reshape(3, 3).T
        object_translation = np.array(object_transformation['object_T']).reshape(1, 3)

        # Apply transformation to object mesh
        object_mesh.vertices = object_mesh_vertices.dot(object_rotation.T) + object_translation
        object_mesh_file = join(save_path_object, f'{str(frame_idx).zfill(6)}.obj')
        object_mesh.export(object_mesh_file, include_texture=False)


def main():
    # Setup device
    device = setup_device()

    # Parse arguments
    args = parse_arguments()

    # Setup paths
    dataset_info_path = join(args.root_path, 'dataset_information.json')
    start_frame_path = join(args.root_path, 'startframe.json')
    mocap_path = join(args.root_path, 'mocap', args.seq_name)
    save_path_human = join(mocap_path, 'obj/human')
    os.makedirs(save_path_human, exist_ok=True)
    save_path_object = join(mocap_path, 'obj/object')
    os.makedirs(save_path_object, exist_ok=True)

    # Load SMPL model
    body_model = load_smpl_model(device)

    # Load object template
    object_mesh, object_mesh_vertices = load_object_template(args.root_path, args.seq_name)

    # Load video files and start frame information
    videos_path = join(args.root_path, 'videos', args.seq_name)
    video_files = sorted(os.listdir(videos_path))
    start_frame = load_start_frame_info(start_frame_path, args.seq_name)

    # Get total frames in the video
    video_path = join(videos_path, video_files[0])
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    # Process frames
    process_frames(start_frame, total_frames, mocap_path, save_path_human, save_path_object, body_model, object_mesh,
                   object_mesh_vertices)


if __name__ == "__main__":
    main()
