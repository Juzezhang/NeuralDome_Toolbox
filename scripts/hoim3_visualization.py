import sys
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
from models.body_models_easymocap.smpl import SMPLModel
from utils.rotation_utils import rot6d_to_matrix
import matplotlib.pyplot as plt
import trimesh

# Set up the computation device based on CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Parse command line arguments
parser = argparse.ArgumentParser(description='hoim3 command line tools')
parser.add_argument('--root_path', type=str, default="/nas/nas_10/AI-being/HOIM3")
parser.add_argument('--seq_name', type=str, default="livingroom_data01")
parser.add_argument('--resolution', type=int, default=720)
parser.add_argument('--step', type=int, default=60)
parser.add_argument('--output_path', type=str, default="/nas/nas_10/AI-being/HOIM3/vis")
parser.add_argument('--vis_view', type=str, default="0")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--end_frame', type=int, default=0)

args = parser.parse_args()

# Setup paths
dataset_info_path = join(args.root_path, 'dataset_information.json')
root_path = join(args.root_path, args.seq_name)
mocap_path = join(args.root_path, 'mocap', args.seq_name, 'object')
mocap_path_human = join(args.root_path, 'mocap', args.seq_name, 'smpl')
seq_step = args.step
body_model = SMPLModel(model_path='models/model_files/smpl/SMPL_NEUTRAL.pkl', device=device)

# Load object template meshes
object_names = os.listdir(mocap_path)
object_mesh_list = []
object_vertex_list = []
for object_name in object_names:
    # Construct the path for the object's template mesh
    object_template_path = join(args.root_path, 'scanned_object', object_name,
                                object_name + '_simplified_transformed.obj')
    object_mesh = trimesh.load(object_template_path, process=False)
    # Center the mesh by subtracting its mean position
    object_mesh.vertices -= object_mesh.vertices.mean(0)
    object_mesh_list.append(object_mesh)

# Load calibration data
with open(dataset_info_path, 'rb') as file:
    dataset_info = json.load(file)
# Determine the calibration path based on the sequence name
calibration_dates = ['20230912', '20230916', '20230924', '20230925', '20230927', '20230928', '20231003', '20231004']
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

# Load IMU data for the first object in the sequence
imu_data_path = join(args.root_path, 'imu_data', args.seq_name, 'imu_data', object_names[0] + '.json')
with open(imu_data_path, 'rb') as f:
    imu_data = json.load(f)
# Determine start and end frames for processing
if args.start_frame == 0:
    start_frame = int(imu_data['start_frame'])
else:
    start_frame = args.start_frame

if args.end_frame == 0:
    end_frame = int(imu_data['end_frame'])
else:
    end_frame = args.end_frame

# Setup paths for video input and output
videos_path = join(args.root_path, 'videos', args.seq_name)
view_index = int(args.vis_view)
video_name = args.vis_view + '.mp4'
output_dir = join(args.output_path, args.seq_name, str(view_index))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f'Rendering image with camera view {view_index}')
scaling_factor = 2160 / args.resolution

# Adjust intrinsic camera matrix based on the desired resolution
K = np.array(camera_params[str(view_index)]['K']).reshape((3, 3))
K /= scaling_factor
K[2, 2] = 1

# Extrinsic camera parameters
RT = np.array(camera_params[str(view_index)]['RT'])
if len(RT) == 16:
    RT = RT.reshape((4, 4))
else:
    RT = RT.reshape((3, 4))

R = RT[:3, :3]
T = RT[:3, 3:].reshape(3)

# Process video frames
video_path = join(videos_path, video_name)
capture = cv2.VideoCapture(video_path)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Process each frame within the specified range
for frame_count in tqdm(range(start_frame, end_frame), desc="Processing frames"):
    # Skip frames based on the specified step
    if frame_count % seq_step != 0:
        continue
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    _, img = capture.read()
    # Resize and adjust the image format
    img = cv2.resize(img[:, :, ::-1], (int(3840 / scaling_factor), int(2160 / scaling_factor)))

    # Load SMPL parameters for the current frame
    smpl_param_path = join(mocap_path_human, f'{str(frame_count).zfill(6)}.json')
    with open(smpl_param_path, 'rb') as file:
        smpl_params = json.load(file)
    # Convert SMPL parameters to NumPy arrays
    smpl_params_list = []
    for smpl_param in smpl_params:
        smpl_params_list.append(
            {key: np.array(value) if isinstance(value, list) else value for key, value in smpl_param.items()})

    # Update human mesh vertices based on SMPL parameters
    out_mesh_list = []
    for smpl_param in smpl_params_list:
        out_mesh_list.append(body_model(**smpl_param))

    human_mesh_list = []
    for out_mesh in out_mesh_list:
        # Create a trimesh object for the human mesh
        human_mesh = trimesh.Trimesh(vertices=out_mesh[0].cpu().numpy(), faces=body_model.faces)
        human_mesh_list.append(human_mesh)

    # Object transformation
    object_mesh_list_show = []
    for object_idx, object_name in enumerate(object_names):
        # Load the object's rotation and translation data for the current frame
        object_RT_path = join(mocap_path, object_name, 'json', f'{str(frame_count).zfill(6)}.json')
        with open(object_RT_path, 'rb') as file:
            object_rotation_refinement = json.load(file)
        # Convert rotation data from 6D to a 3x3 matrix
        object_R_refinement = np.array(object_rotation_refinement['object_R'])
        object_R_refinement = torch.from_numpy(object_R_refinement)
        object_R_refinement = rot6d_to_matrix(object_R_refinement).numpy().reshape(3, 3).T
        object_T_refinement = np.array(object_rotation_refinement['object_T']).reshape(1, 3)
        # Apply transformation to the object mesh vertices
        object_mesh_vertices = object_mesh_list[object_idx]
        object_mesh_vertices_transformed = object_mesh_vertices.vertices.dot(
            object_R_refinement.T) + object_T_refinement
        object_mesh = trimesh.Trimesh(vertices=object_mesh_vertices_transformed, faces=object_mesh_vertices.faces)
        object_mesh_list_show.append(object_mesh)

    # Initialize 3D visualization wrapper
    pyt3d_wrapper = Pyt3DWrapper(image_size=(int(args.resolution), int(args.resolution * 3840 / 2160)), K=K, R=R, T=T,
                                 image=img)
    # Render meshes
    rendered_image = pyt3d_wrapper.render_meshes(human_mesh_list + object_mesh_list_show, False, R=R, T=T)
    rendered_image[rendered_image > 1] = 1
    image_compare = np.hstack([(rendered_image[:, :, ::-1] * 255).astype(np.uint8), img[:, :, ::-1]])

    # Overlay the frame count on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 255, 0)  # Green color
    thickness = 2
    position = (100, 100)  # Text position
    cv2.putText(image_compare, f'Frame: {frame_count}', position, font, font_scale, color, thickness)

    # Save the rendered image
    save_path = join(output_dir, f'{str(frame_count).zfill(6)}.jpg')
    cv2.imwrite(save_path, image_compare)
    # Optional: display the rendered image
    # plt.imshow(image_compare[:, :, ::-1])
    # plt.show()

capture.release()