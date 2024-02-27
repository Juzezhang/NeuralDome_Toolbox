import sys
sys.path.append('./')
import os
from os.path import join
import json
import cv2
import numpy as np
import torch
import argparse
from tqdm import tqdm
# Importing custom modules for 3D visualization and SMPL model handling
from viz.pyt3d_wrapper import Pyt3DWrapper
from psbody.mesh import Mesh
from models.body_models_easymocap.smplx import SMPLHModel
from yacs.config import CfgNode
from utils.rotation_utils import rot6d_to_matrix

# Set up the computation device based on CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Parse command line arguments
parser = argparse.ArgumentParser(description='NeuralDome command line tools')
parser.add_argument('--root_path', type=str, default="/nas/nas_10/NeuralDome/Hodome")
parser.add_argument('--seq_name', type=str, default="subject01_baseball")
parser.add_argument('--resolution', type=int, default=720)
parser.add_argument('--output_path', type=str, default="/nas/nas_10/NeuralDome/Hodome/vis/")
args = parser.parse_args()

# Setup paths
dataset_info_path = join(args.root_path, 'dataset_information.json')
start_frame_path = join(args.root_path, 'startframe.json')
mocap_path = join(args.root_path, 'mocap', args.seq_name)

# Configuration for SMPL model
cfg_hand = CfgNode()
cfg_hand.use_pca = True
cfg_hand.use_flat_mean = False
cfg_hand.num_pca_comps = 12
body_model = SMPLHModel(model_path='models/model_files/smplhv1.2/neutral/model.npz',
                        device=device, mano_path='models/model_files/manov1.2',
                        cfg_hand=cfg_hand, NUM_SHAPES=16)
human_mesh = Mesh(f=body_model.faces)

# Load object template
object_name = args.seq_name.split('_')[-1]
object_template_path = join(args.root_path, 'scaned_object', object_name, f'{object_name}_face1000.obj')
object_mesh = Mesh()
object_mesh.load_from_file(object_template_path)
object_mesh_vertices = object_mesh.v

# Load calibration data
with open(dataset_info_path, 'rb') as file:
    dataset_info = json.load(file)
# Determine the calibration path based on the sequence name
calibration_dates = ['20221018', '20221019', '20221020']
calibration_path = ''
for date in calibration_dates:
    if args.seq_name in dataset_info[date]:
        calibration_path = join(args.root_path, 'calibration', date, 'calibration.json')
        break
if not calibration_path:
    sys.exit("Calibration data not found for the given sequence name.")

# Load camera parameters
with open(calibration_path, 'rb') as file:
    camera_params = json.load(file)

videos_path = join(args.root_path, 'videos', args.seq_name)
video_files = sorted(os.listdir(videos_path))

# Load start frame information
with open(start_frame_path, 'rb') as file:
    start_frames = json.load(file)

# Process each video
for video_name in video_files:
    view_index = int(video_name.split('.')[0][4:]) - 1
    output_dir = join(args.output_path, args.seq_name, str(view_index))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Rendering image with camera view {view_index}')
    scaling_factor = 2160 / args.resolution
    # Intrinsic camera matrix adjustment
    K = np.array(camera_params[str(view_index)]['K']).reshape((3, 3))
    K /= scaling_factor
    K[2, 2] = 1
    # Extrinsic camera parameters
    RT = np.array(camera_params[str(view_index)]['RT']).reshape((4, 4))
    R = RT[:3, :3]
    T = RT[:3, 3:].reshape(3)
    # Process video frames
    video_path = join(videos_path, video_name)
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = start_frames['rgb'][args.seq_name]
    for frame_count in tqdm(range(start_frame, total_frames), desc="Processing frames"):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        _, img = capture.read()
        img = cv2.resize(img[:, :, ::-1], (int(3840 / scaling_factor), int(2160 / scaling_factor)))
        # Load SMPL parameters for the current frame
        smpl_param_path = join(mocap_path, 'smplh', f'{str(frame_count).zfill(6)}.json')
        with open(smpl_param_path, 'rb') as file:
            smpl_params = json.load(file)['annots'][0]
        # Convert SMPL parameters to NumPy arrays
        smpl_params = {key: np.array(value) if isinstance(value, list) else value for key, value in smpl_params.items()}
        # Update human mesh vertices
        out_mesh = body_model(**smpl_params)
        human_mesh.v = out_mesh[0].cpu().numpy()

        # Object transformation
        object_RT_path = join(mocap_path, 'object/refine/json', f'{str(frame_count).zfill(6)}.json')
        with open(object_RT_path, 'rb') as file:
            object_rotation_refinement = json.load(file)
        object_R_refinement = np.array(object_rotation_refinement['object_R'])
        object_R_refinement = torch.from_numpy(object_R_refinement)
        object_R_refinement = rot6d_to_matrix(object_R_refinement).numpy().reshape(3, 3).T
        object_T_refinement = np.array(object_rotation_refinement['object_T']).reshape(1, 3)
        # Apply transformation to the object mesh vertices
        object_mesh.v = object_mesh_vertices.dot(object_R_refinement.T) + object_T_refinement
        # Initialize 3D visualization wrapper
        pyt3d_wrapper = Pyt3DWrapper(image_size=(720, 1280), K=K, R=R, T=T, image=img)
        # Render meshes
        rendered_image = pyt3d_wrapper.render_meshes([human_mesh, object_mesh], False, R=R, T=T)
        # Save the rendered image
        save_path = join(output_dir, f'{str(frame_count).zfill(6)}.jpg')
        cv2.imwrite(save_path, 255 * rendered_image[:, :, ::-1])
        # Optional: display the rendered image
        # plt.imshow(rendered_image)
        # plt.show()
    capture.release()
