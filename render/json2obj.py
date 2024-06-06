import argparse
import glob
import json
import os
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from tqdm import tqdm

from body_model.body_model import BodyModel


def load_file(filename):
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    elif filename.endswith('.json'):
        with open(filename, 'rb') as file:
            data = json.load(file)
    return data


def rot6d_to_matrix(rot_6d):
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def process_hodome(obj_subjects, person_subjects, args, start):
    for i, subject in enumerate(tqdm(obj_subjects)):
        batch_size = args.end
        # Prepare the body_model
        body_model = BodyModel('/data/MPMO/body_model/smplh/neutral/model.npz',
                               num_betas=16,
                               batch_size=batch_size,
                               num_expressions=None,
                               model_type='smplh').to(args.device)
        body_model_one = BodyModel('/data/MPMO/body_model/smplh/neutral/model.npz',
                                   num_betas=16,
                                   batch_size=1,
                                   num_expressions=None,
                                   model_type='smplh').to(args.device)
        pose_body, bdata_trans, Rh, betas, pose_hand, objectRot, objectTrans, objectRot_matrix = [], [], [], [], [], [], [], []
        for frame in range(batch_size):
            obj_motion = load_file(obj_subjects[i][0][frame])
            person_motion = load_file(person_subjects[i][0][frame])['annots'][0]
            body_parms_one = {
                'root_orient': np.zeros([1, 3]),
                'pose_body': np.array(person_motion["poses"][0]).reshape(1, -1)[:, 3:66],
                'trans': np.zeros([1, 3]),
                'betas': np.array(person_motion["shapes"][0]).reshape(1, -1),
                'pose_hand': np.array(person_motion["poses"][0]).reshape(1, -1)[:, 66:]
            }
            body_pose_one = body_model_one(
                **{k: torch.tensor(v).float().to(args.device) for k, v in body_parms_one.items() if
                   k in ['pose_body', 'root_orient', 'trans', 'betas', 'pose_hand']})
            j0 = body_pose_one.Jtr[:, 0, :].cpu()
            rot = axis_angle_to_matrix(torch.tensor(person_motion["Rh"][0]).reshape(1, -1))
            Th = torch.tensor(person_motion["Th"][0]).reshape(1, -1)
            Tnew = Th - j0 + torch.einsum('bij,bj->bi', rot, j0)

            # Human
            pose_body.append(np.array(person_motion["poses"][0]).reshape(1, -1)[:, :66])
            Rh.append(np.array(person_motion["Rh"][0]).reshape(1, -1))
            bdata_trans.append(np.array(Tnew))
            betas.append(np.array(person_motion["shapes"][0]).reshape(1, -1))
            pose_hand.append(np.array(person_motion["poses"][0]).reshape(1, -1)[:, 66:])
            # Object
            O_R = torch.tensor(obj_motion["object_R"])
            O_R_matrix = rot6d_to_matrix(O_R)[0].T.unsqueeze(0)
            O_R_aa = matrix_to_axis_angle(O_R_matrix).numpy()
            objectRot_matrix.append(O_R_matrix.numpy())
            objectRot.append(O_R_aa)
            objectTrans.append(np.array(obj_motion["object_T"]).reshape(1, -1))

        pose_body = np.stack(pose_body).squeeze()
        Rh = np.stack(Rh).squeeze()
        bdata_trans = np.stack(bdata_trans).squeeze()
        betas = np.stack(betas).squeeze()
        pose_hand = np.stack(pose_hand).squeeze()
        objectTrans = np.stack(objectTrans).squeeze()
        objectRot_matrix = np.stack(objectRot_matrix).squeeze()
        # Loading the object template
        verts, faces_idx, _ = load_obj(join(args.object_root, args.seq.split('_')[-1], args.seq.split('_')[-1] + '_face1000.obj'))
        # Normalize the object template and rotation & translation
        verts_temps = []
        verts_temp_all = []
        for frame in range(batch_size):
            verts_temp = torch.matmul(verts, torch.tensor(objectRot_matrix[frame].T).float()) + \
                         torch.tensor(objectTrans)[frame].float()
            verts_temp_all.append(verts_temp)
            verts_temps.append(verts_temp.mean(0).reshape(1, -1))

        body_parms = {
            'root_orient': Rh[:, :3],
            'pose_body': pose_body[:, 3:66],
            'trans': bdata_trans,
            'betas': betas,
            'pose_hand': pose_hand
        }
        body_pose_world = body_model(**{k: torch.tensor(v).float().to(args.device) for k, v in body_parms.items() if
                                        k in ['pose_body', 'root_orient', 'trans', 'betas', 'pose_hand']})

        save_path = os.path.join(args.file_root, args.seq, 'vis/obj')
        os.makedirs(save_path, exist_ok=True)

        for h in tqdm(range(batch_size)):
            save_obj(join(save_path, "{:06d}.obj".format(h + start)), verts=verts_temp_all[h],
                     faces=faces_idx.verts_idx)
            save_obj(join(save_path, "{:06d}_h.obj".format(h + start)), verts=body_pose_world.v[h],
                     faces=body_pose_world.f)


def main(args):
    obj_subjects, person_subjects = [], []
    data = join(args.file_root, args.seq)
    start = int(sorted(glob.glob(os.path.join(data, 'object/refine/json/*.json')))[0].split('/')[-1].split('.')[0])
    obj_subject = [sorted(glob.glob(os.path.join(data, 'object/refine/json/*.json')))]
    obj_subjects.append(obj_subject)
    person_subject = [sorted(glob.glob(os.path.join(data, 'smplh/*.json')))[start:]]
    person_subjects.append(person_subject)
    process_hodome(obj_subjects, person_subjects, args, start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_root', type=str, default='/nas/nas_10/NeuralDome/Hodome/mocap/')
    parser.add_argument('--object_root', type=str, default='/nas/nas_10/NeuralDome/Hodome/scaned_object/')
    parser.add_argument('--seq', type=str, default='subject02_desk')
    parser.add_argument('--end', type=int, default='5')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)