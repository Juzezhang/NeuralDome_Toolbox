# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from smplx import SMPL as _SMPL
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

from ikol.core.conventions.keypoints_mapping import (
    convert_kps,
    get_keypoint_num,
)
from ikol.core.conventions.segmentation import body_segmentation
# from hybrik.models.utils.inverse_kinematics import batch_inverse_kinematics_transform, batch_inverse_kinematics_transform_train,batch_inverse_kinematics_transform_optimized, batch_get_pelvis_orient_v2, batch_get_pelvis_orient_svd_v2, rotmat_to_quat
from ikol.utils.transforms import quat_to_rotmat

try:
    import cPickle as pk
except ImportError:
    import pickle as pk



ModelOutput = namedtuple('ModelOutput',
                         ['vertices', 'joints', 'joints_from_verts',
                          'rot_mats'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)




class SMPL(_SMPL):
    """Extension of the official SMPL implementation."""

    body_pose_keys = {
        'global_orient',
        'body_pose',
    }
    full_pose_keys = {
        'global_orient',
        'body_pose',
    }
    NUM_VERTS = 6890
    NUM_FACES = 13776

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smpl_45',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs) -> None:
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        self.keypoint_src = keypoint_src
        self.keypoint_dst = keypoint_dst
        self.keypoint_approximate = keypoint_approximate
        # override the default SMPL joint regressor if available
        if joints_regressor is not None:
            joints_regressor = torch.tensor(
                np.load(joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor', joints_regressor)

        # allow for extra joints to be regressed if available
        if extra_joints_regressor is not None:
            joints_regressor_extra = torch.tensor(
                np.load(extra_joints_regressor), dtype=torch.float)
            self.register_buffer('joints_regressor_extra',
                                 joints_regressor_extra)

        self.num_verts = self.get_num_verts()
        self.num_joints = get_keypoint_num(convention=self.keypoint_dst)
        self.body_part_segmentation = body_segmentation('smpl')

    def forward(self,
                *args,
                return_verts: bool = True,
                return_full_pose: bool = False,
                **kwargs) -> dict:
        """Forward function.

        Args:
            *args: extra arguments for SMPL
            return_verts: whether to return vertices
            return_full_pose: whether to return full pose parameters
            **kwargs: extra arguments for SMPL

        Returns:
            output: contains output parameters and attributes
        """

        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        if not hasattr(self, 'joints_regressor'):
            joints = smpl_output.joints
        else:
            joints = vertices2joints(self.joints_regressor,
                                     smpl_output.vertices)

        if hasattr(self, 'joints_regressor_extra'):
            extra_joints = vertices2joints(self.joints_regressor_extra,
                                           smpl_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        joints, joint_mask = convert_kps(
            joints,
            src=self.keypoint_src,
            dst=self.keypoint_dst,
            approximate=self.keypoint_approximate)
        if isinstance(joint_mask, np.ndarray):
            joint_mask = torch.tensor(
                joint_mask, dtype=torch.uint8, device=joints.device)

        batch_size = joints.shape[0]
        joint_mask = joint_mask.reshape(1, -1).expand(batch_size, -1)

        output = dict(
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            joint_mask=joint_mask,
            keypoints=torch.cat([joints, joint_mask[:, :, None]], dim=-1),
            betas=smpl_output.betas)

        if return_verts:
            output['vertices'] = smpl_output.vertices
        if return_full_pose:
            output['full_pose'] = smpl_output.full_pose

        return output

    @classmethod
    def tensor2dict(cls,
                    full_pose: torch.Tensor,
                    betas: Optional[torch.Tensor] = None,
                    transl: Optional[torch.Tensor] = None):
        """Convert full pose tensor to pose dict.

        Args:
            full_pose (torch.Tensor): shape should be (..., 165) or
                (..., 55, 3). All zeros for T-pose.
            betas (Optional[torch.Tensor], optional): shape should be
                (..., 10). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
            transl (Optional[torch.Tensor], optional): shape should be
                (..., 3). The batch num should be 1 or corresponds with
                full_pose.
                Defaults to None.
        Returns:
            dict: dict of smpl pose containing transl & betas.
        """
        full_pose = full_pose.view(-1, (cls.NUM_BODY_JOINTS + 1) * 3)
        body_pose = full_pose[:, 3:]
        global_orient = full_pose[:, :3]
        batch_size = full_pose.shape[0]
        if betas is not None:
            # squeeze or unsqueeze betas to 2 dims
            betas = betas.view(-1, betas.shape[-1])
            if betas.shape[0] == 1:
                betas = betas.repeat(batch_size, 1)
        else:
            betas = betas
        transl = transl.view(batch_size, -1) if transl is not None else transl
        return {
            'betas': betas,
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
        }

    @classmethod
    def dict2tensor(cls, smpl_dict: dict) -> torch.Tensor:
        """Convert smpl pose dict to full pose tensor.

        Args:
            smpl_dict (dict): smpl pose dict.

        Returns:
            torch: full pose tensor.
        """
        assert cls.body_pose_keys.issubset(smpl_dict)
        for k in smpl_dict:
            if isinstance(smpl_dict[k], np.ndarray):
                smpl_dict[k] = torch.Tensor(smpl_dict[k])
        global_orient = smpl_dict['global_orient'].view(-1, 3)
        body_pose = smpl_dict['body_pose'].view(-1, 3 * cls.NUM_BODY_JOINTS)
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        return full_pose


class GenderedSMPL(torch.nn.Module):
    """A wrapper of SMPL to handle gendered inputs."""

    def __init__(self,
                 *args,
                 keypoint_src: str = 'smpl_45',
                 keypoint_dst: str = 'human_data',
                 keypoint_approximate: bool = False,
                 joints_regressor: str = None,
                 extra_joints_regressor: str = None,
                 **kwargs) -> None:
        """
        Args:
            *args: extra arguments for SMPL initialization.
            keypoint_src: source convention of keypoints. This convention
                is used for keypoints obtained from joint regressors.
                Keypoints then undergo conversion into keypoint_dst
                convention.
            keypoint_dst: destination convention of keypoints. This convention
                is used for keypoints in the output.
            keypoint_approximate: whether to use approximate matching in
                convention conversion for keypoints.
            joints_regressor: path to joint regressor. Should be a .npy
                file. If provided, replaces the official J_regressor of SMPL.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(GenderedSMPL, self).__init__()

        assert 'gender' not in kwargs, \
            self.__class__.__name__ + \
            'does not need \'gender\' for initialization.'

        self.smpl_neutral = SMPL(
            *args,
            gender='neutral',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_male = SMPL(
            *args,
            gender='male',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.smpl_female = SMPL(
            *args,
            gender='female',
            keypoint_src=keypoint_src,
            keypoint_dst=keypoint_dst,
            keypoint_approximate=keypoint_approximate,
            joints_regressor=joints_regressor,
            extra_joints_regressor=extra_joints_regressor,
            **kwargs)

        self.num_verts = self.smpl_neutral.num_verts
        self.num_joints = self.smpl_neutral.num_joints
        self.faces = self.smpl_neutral.faces

    def forward(self,
                *args,
                betas: torch.Tensor = None,
                body_pose: torch.Tensor = None,
                global_orient: torch.Tensor = None,
                transl: torch.Tensor = None,
                return_verts: bool = True,
                return_full_pose: bool = False,
                gender: torch.Tensor = None,
                device=None,
                **kwargs):
        """Forward function.
        Note:
            B: batch size
            J: number of joints of model, J = 23 (SMPL)
            K: number of keypoints
        Args:
            *args: extra arguments
            betas: Tensor([B, 10]), human body shape parameters of SMPL model.
            body_pose: Tensor([B, J*3] or [B, J, 3, 3]), human body pose
                parameters of SMPL model. It should be axis-angle vector
                ([B, J*3]) or rotation matrix ([B, J, 3, 3)].
            global_orient: Tensor([B, 3] or [B, 1, 3, 3]), global orientation
                of human body. It should be axis-angle vector ([B, 3]) or
                rotation matrix ([B, 1, 3, 3)].
            transl: Tensor([B, 3]), global translation of human body.
            gender: Tensor([B]), gender parameters of human body. -1 for
                neutral, 0 for male , 1 for female.
            device: the device of the output
            **kwargs: extra keyword arguments
        Returns:
            outputs (dict): Dict with mesh vertices and joints.
                - vertices: Tensor([B, V, 3]), mesh vertices
                - joints: Tensor([B, K, 3]), 3d keypoints regressed from
                    mesh vertices.
        """

        batch_size = None
        for attr in [betas, body_pose, global_orient, transl]:
            if attr is not None:
                if device is None:
                    device = attr.device
                if batch_size is None:
                    batch_size = attr.shape[0]
                else:
                    assert batch_size == attr.shape[0]

        if gender is not None:
            output = {
                'vertices':
                torch.zeros([batch_size, self.num_verts, 3], device=device),
                'joints':
                torch.zeros([batch_size, self.num_joints, 3], device=device),
                'joint_mask':
                torch.zeros([batch_size, self.num_joints],
                            dtype=torch.uint8,
                            device=device)
            }

            for body_model, gender_label in \
                    [(self.smpl_neutral, -1),
                     (self.smpl_male, 0),
                     (self.smpl_female, 1)]:
                gender_idxs = gender == gender_label

                # skip if no such gender is present
                if gender_idxs.sum() == 0:
                    continue

                output_model = body_model(
                    betas=betas[gender_idxs] if betas is not None else None,
                    body_pose=body_pose[gender_idxs]
                    if body_pose is not None else None,
                    global_orient=global_orient[gender_idxs]
                    if global_orient is not None else None,
                    transl=transl[gender_idxs] if transl is not None else None,
                    **kwargs)

                output['joints'][gender_idxs] = output_model['joints']

                # TODO: quick fix
                if 'joint_mask' in output_model:
                    output['joint_mask'][gender_idxs] = output_model[
                        'joint_mask']

                if return_verts:
                    output['vertices'][gender_idxs] = output_model['vertices']
                if return_full_pose:
                    output['full_pose'][gender_idxs] = output_model[
                        'full_pose']
        else:
            output = self.smpl_neutral(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                **kwargs)

        return output


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class HybrIKSMPL(SMPL):
    """Extension of the SMPL for HybrIK."""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',  # 2
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',  # 8
        'spine3',
        'left_foot',
        'right_foot',  # 11
        'neck',
        'left_collar',
        'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder',
        'right_shoulder',  # 17
        'left_elbow',
        'right_elbow',  # 19
        'left_wrist',
        'right_wrist',  # 21
        'left_thumb',
        'right_thumb',  # 23
        'head',
        'left_middle',
        'right_middle',  # 26
        'left_bigtoe',
        'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, *args, extra_joints_regressor=None, **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(HybrIKSMPL, self).__init__(
            *args,
            extra_joints_regressor=extra_joints_regressor,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            **kwargs)

        self.dtype = torch.float32
        self.num_joints = 29
        # self.num_joints = 24
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [
            self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES
        ]
        self.SPINE3_IDX = 9
        # # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        # extend kinematic tree
        parents[:24] = self.parents
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map',
                             self._parents_to_children(parents))
        self.parents = parents

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_skeleton,
                betas,
                phis,
                global_orient,
                transl=None,
                return_verts=True,
                leaf_thetas=None):
        """Inverse pass for the SMPL model.

        Args:
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (img, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            phis: torch.tensor, shape Bx23x2
                Rotation on bone axis parameters
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)
            leaf_thetas: torch.tensor, optional, shape Bx5x4
                Quaternions of 5 leaf joints. (default=None)

        Returns
            outputs: output dictionary.
        """
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        batch_size = max(betas.shape[0], pose_skeleton.shape[0])
        device = betas.device

        # 1. Add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        # 2. Get the rest joints
        # NxJx3 array
        if leaf_thetas is not None:
            rest_J = vertices2joints(self.J_regressor, v_shaped)
        else:
            rest_J = torch.zeros((v_shaped.shape[0], 29, 3),
                                 dtype=self.dtype,
                                 device=device)
            rest_J[:, :24] = vertices2joints(self.J_regressor, v_shaped)

            leaf_number = [411, 2445, 5905, 3216, 6617]
            leaf_vertices = v_shaped[:, leaf_number].clone()
            rest_J[:, 24:] = leaf_vertices

        # 3. Get the rotation matrics
        rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
            pose_skeleton,
            global_orient,
            phis,
            rest_J.clone(),
            self.children_map,
            self.parents,
            dtype=self.dtype,
            train=self.training,
            leaf_thetas=leaf_thetas)

        test_joints = True
        if test_joints:
            new_joints, A = batch_rigid_transform(
                rot_mats,
                rest_J[:, :24].clone(),
                self.parents[:24],
                dtype=self.dtype)
        else:
            new_joints = None

        # assert torch.mean(torch.abs(rotate_rest_pose - new_joints)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype,
                                   device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.joints_regressor_extra,
                                            vertices)

        # rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - \
                new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - \
                joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = {
            'vertices': vertices,
            'joints': new_joints,
            'poses': rot_mats,
            'joints_from_verts': joints_from_verts,
        }
        return output


class HybrIK24SMPL(SMPL):
    """Extension of the SMPL for HybrIK."""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',  # 2
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',  # 8
        'spine3',
        'left_foot',
        'right_foot',  # 11
        'neck',
        'left_collar',
        'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder',
        'right_shoulder',  # 17
        'left_elbow',
        'right_elbow',  # 19
        'left_wrist',
        'right_wrist',  # 21
        'left_thumb',
        'right_thumb',  # 23
        'head',
        'left_middle',
        'right_middle',  # 26
        'left_bigtoe',
        'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, *args, extra_joints_regressor=None, **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(HybrIK24SMPL, self).__init__(
            *args,
            extra_joints_regressor=extra_joints_regressor,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            **kwargs)

        self.dtype = torch.float32
        # self.num_joints = 29
        self.num_joints = 24
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [
            self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES
        ]
        self.SPINE3_IDX = 9
        # # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        # extend kinematic tree
        parents[:24] = self.parents
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map',
                             self._parents_to_children(parents))
        self.parents = parents

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_skeleton,
                betas,
                phis,
                global_orient,
                transl=None,
                return_verts=True,
                leaf_thetas=None):
        """Inverse pass for the SMPL model.

        Args:
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (img, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            phis: torch.tensor, shape Bx23x2
                Rotation on bone axis parameters
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)
            leaf_thetas: torch.tensor, optional, shape Bx5x4
                Quaternions of 5 leaf joints. (default=None)

        Returns
            outputs: output dictionary.
        """
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        batch_size = max(betas.shape[0], pose_skeleton.shape[0])
        device = betas.device

        # 1. Add shape contribution
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)

        # 2. Get the rest joints
        # NxJx3 array
        if leaf_thetas is not None:
            rest_J = vertices2joints(self.J_regressor, v_shaped)
        else:
            rest_J = torch.zeros((v_shaped.shape[0], 29, 3),
                                 dtype=self.dtype,
                                 device=device)
            rest_J[:, :24] = vertices2joints(self.J_regressor, v_shaped)

            leaf_number = [411, 2445, 5905, 3216, 6617]
            leaf_vertices = v_shaped[:, leaf_number].clone()
            rest_J[:, 24:] = leaf_vertices

        # 3. Get the rotation matrics
        # rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
        #     pose_skeleton,
        #     global_orient,
        #     phis,
        #     rest_J.clone(),
        #     self.children_map,
        #     self.parents,
        #     dtype=self.dtype,
        #     train=self.training,
        #     leaf_thetas=leaf_thetas)

        if self.training:
            rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_train(
                pose_skeleton, global_orient, phis,
                rest_J.clone(), self.children_map, self.parents, dtype=self.dtype, train=self.training,
                leaf_thetas=leaf_thetas)
        else:
            rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_optimized(
                pose_skeleton, phis,
                rest_J.clone(), self.children_map, self.parents, dtype=self.dtype, train=self.training,
                leaf_thetas=leaf_thetas)


        test_joints = True
        if test_joints:
            new_joints, A = batch_rigid_transform(
                rot_mats,
                rest_J[:, :24].clone(),
                self.parents[:24],
                dtype=self.dtype)
        else:
            new_joints = None

        # assert torch.mean(torch.abs(rotate_rest_pose - new_joints)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype,
                                   device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.joints_regressor_extra,
                                            vertices)

        # rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - \
                new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - \
                joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = {
            'vertices': vertices,
            'joints': new_joints,
            'poses': rot_mats,
            'joints_from_verts': joints_from_verts,
        }
        return output


class HybrIKOptSMPL(SMPL):
    """Extension of the SMPL for HybrIK."""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',  # 2
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',  # 8
        'spine3',
        'left_foot',
        'right_foot',  # 11
        'neck',
        'left_collar',
        'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder',
        'right_shoulder',  # 17
        'left_elbow',
        'right_elbow',  # 19
        'left_wrist',
        'right_wrist',  # 21
        'left_thumb',
        'right_thumb',  # 23
        'head',
        'left_middle',
        'right_middle',  # 26
        'left_bigtoe',
        'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, *args, extra_joints_regressor=None, **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(HybrIKOptSMPL, self).__init__(
            *args,
            extra_joints_regressor=extra_joints_regressor,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            **kwargs)

        self.dtype = torch.float32
        # self.num_joints = 29
        self.num_joints = 24
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [
            self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES
        ]
        self.SPINE3_IDX = 9
        # # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        # extend kinematic tree
        parents[:24] = self.parents
        # parents[24] = 15
        # parents[25] = 22
        # parents[26] = 23
        # parents[27] = 10
        # parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map',
                             self._parents_to_children(parents))
        self.register_buffer(
            'children_map_opt',
            self._parents_to_children_opt(parents))


        self.parents = parents
        self.idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [20, 21],  # 7
            [15, 22, 23, 10, 11]  # 8
            ]

        self.parent_indexs = [
            [-1],  # 0
            [-1],  # 1
            [-1],  # 2
            [-1],  # 3
            [0, 1],  # 4
            [0, 2],  # 5
            [0, 3],  # 6
            [0, 1, 4],  # 7
            [0, 2, 5],  # 8
            [0, 3, 6],  # 9
            [0, 1, 4, 7],  # 10
            [0, 2, 5, 8],  # 11
            [0, 3, 6, 9],  # 12
            [0, 3, 6, 9],  # 13
            [0, 3, 6, 9],  # 14
            [0, 3, 6, 9, 12],  # 15
            [0, 3, 6, 9, 13],  # 16
            [0, 3, 6, 9, 14],  # 17
            [0, 3, 6, 9, 13, 16],  # 18
            [0, 3, 6, 9, 14, 17],  # 19
            [0, 3, 6, 9, 13, 16, 18],  # 20
            [0, 3, 6, 9, 14, 17, 19],  # 21
            [0, 3, 6, 9, 13, 16, 18, 20],  # 22
            [0, 3, 6, 9, 14, 17, 19, 21]  # 23
            ]  # 受到影响的父节点 index

        self.idx_jacobian = [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19],
            [20, 21],
            [22, 23]
        ]  # 少了0,1,2,3

        self.index_18_to_24 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    def _parents_to_children_opt(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        # children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[-1] = -1
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children


    def forward_global_orient(self,
                pose_skeleton,
                rest_J):


        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()  # 防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # TODO
        if self.training:
            global_orient_mat = batch_get_pelvis_orient_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)
        else:
            global_orient_mat = batch_get_pelvis_orient_svd_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)

        return global_orient_mat


    def forward_twist_and_leaf_train(self,
                rest_J,
                phis,
                global_orient,
                leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        rot_mat_local = [global_orient]
        # leaf nodes rot_mats
        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            rotmat_leaf = quat_to_rotmat(leaf_thetas)
            rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])
        else:
            rotmat_leaf_ = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(batch_size, 5, 1, 1)
            # rotmat_leaf_ = None


        for indices in range(1, 24):

            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
            ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)

            # (B, K, 1, 1)
            cos, sin = torch.split(phis[:, indices - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)

            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat_local.append(rot_mat_spin)

        # (B, K + 1, 3, 3)
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return rot_mats, rotmat_leaf_

    def forward_jacobian_and_pred_train(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # rot_mat_chain[:, parents_1] = torch.matmul(
            #     rot_mat_chain[:, parents_2],
            #     rot_mat)
            # rot_mat_local[:, parents_1] = rot_mat

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw, rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]

                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @  DR_k_1_k_1

                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_1)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_2]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_2]
                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_2])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]])
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]

                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()
                    # rot_mat_withDR2 = rot_mat_local[0].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @ DR_k_1_k_2
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]] @ DR_k_2_k_2 @ rot_mat_local[parent_index[-2]]

                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_2)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-3]:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_2)
                    #     else:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_3]
                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_3])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    # Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Temp_q = rot_mat_withDR0 + torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]])
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()
                    # rot_mat_withDR2 = rot_mat_local[0].clone()
                    # rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @ DR_k_1_k_3
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]] @ DR_k_2_k_3 @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]] @ DR_k_3_k_3 @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]

                    #
                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_3)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-3]:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_3)
                    #     else:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-4]:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_3)
                    #     else:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_4]
                    #
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_4]
                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_4])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    # rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]])

                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]
                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()
                    # rot_mat_withDR2 = rot_mat_local[0].clone()
                    # rot_mat_withDR3 = rot_mat_local[0].clone()
                    # rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @ DR_k_1_k_4
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]] @ DR_k_2_k_4 @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]] @ DR_k_3_k_4 @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]] @ DR_k_4_k_4 @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]

                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_4)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-3]:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_4)
                    #     else:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-4]:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_4)
                    #     else:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-5]:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_4)
                    #     else:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_5]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_5]

                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_5])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    # rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    # rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    # Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]])

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]

                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()
                    # rot_mat_withDR2 = rot_mat_local[0].clone()
                    # rot_mat_withDR3 = rot_mat_local[0].clone()
                    # rot_mat_withDR4 = rot_mat_local[0].clone()
                    # rot_mat_withDR5 = rot_mat_local[0].clone()



                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @ DR_k_1_k_5
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]] @ DR_k_2_k_5 @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]] @ DR_k_3_k_5 @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]] @ DR_k_4_k_5 @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR5 = rot_mat_chain[parent_index[-7]] @ DR_k_5_k_5 @ rot_mat_local[parent_index[-5]] @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]


                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_5)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-3]:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_5)
                    #     else:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-4]:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_5)
                    #     else:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-5]:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_5)
                    #     else:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    #
                    #     if index_r == parent_index[-6]:
                    #         rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_5)
                    #     else:
                    #         rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_6]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_6]

                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_6])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    # rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    # rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    # rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    # Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR5, rot_mat_local[parent_index[-1]])


                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    # rot_mat_withDR0 = rot_mat_local[0].clone()
                    # rot_mat_withDR1 = rot_mat_local[0].clone()
                    # rot_mat_withDR2 = rot_mat_local[0].clone()
                    # rot_mat_withDR3 = rot_mat_local[0].clone()
                    # rot_mat_withDR4 = rot_mat_local[0].clone()
                    # rot_mat_withDR5 = rot_mat_local[0].clone()
                    # rot_mat_withDR6 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]] @ DR_k_1_k_6
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]] @ DR_k_2_k_6 @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]] @ DR_k_3_k_6 @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]] @ DR_k_4_k_6 @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR5 = rot_mat_chain[parent_index[-7]] @ DR_k_5_k_6 @ rot_mat_local[parent_index[-5]] @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]
                    rot_mat_withDR6 = rot_mat_chain[parent_index[-8]] @ DR_k_6_k_6 @ rot_mat_local[parent_index[-6]] @ rot_mat_local[parent_index[-5]] @ rot_mat_local[parent_index[-4]] @ rot_mat_local[parent_index[-3]] @ rot_mat_local[parent_index[-2]]

                    # for index_r in parent_index[1:-1]:
                    #     if index_r == parent_index[-2]:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_6)
                    #     else:
                    #         rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    #     if index_r == parent_index[-3]:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_6)
                    #     else:
                    #         rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    #     if index_r == parent_index[-4]:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_6)
                    #     else:
                    #         rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    #     if index_r == parent_index[-5]:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_6)
                    #     else:
                    #         rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    #     if index_r == parent_index[-6]:
                    #         rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_6)
                    #     else:
                    #         rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                    #     if index_r == parent_index[-7]:
                    #         rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, DR_k_6_k_6)
                    #     else:
                    #         rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]] @ R_derivative[:, parents_1, parents_7]

                    # for index_r in parent_index[1:]:
                    #     if index_r == parent_index[-1]:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_7])
                    #     else:
                    #         rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    # rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    # rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    # rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                    # rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])

                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR5, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR6, rot_mat_local[parent_index[-1]])

                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats

    def forward_jacobian_and_pred(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # rot_mat_chain[:, parents_1] = torch.matmul(
            #     rot_mat_chain[:, parents_2],
            #     rot_mat)
            # rot_mat_local[:, parents_1] = rot_mat

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    # rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    # rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    # rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                    # rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                    # for index_r in range(1, tree_len - 1):
                    #     rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                    #                                    rot_mat_local_withDR1[:, index_r])

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_1)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_2]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_2])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_2)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_2)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_3])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_3)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_3)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_3)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_4]
                    #
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_4])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_4)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_4)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_4)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_4)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_5]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_5])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_5)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_5)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_5)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_5)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_5)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_6]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_6])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    rot_mat_withDR6 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_6)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_6)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_6)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_6)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_6)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                        if index_r == parent_index[-7]:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, DR_k_6_k_6)
                        else:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_7])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats

    def single_iteration_train(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         global_orient,
                         rot_mat_twist=None,
                         rotmat_leaf=None,
                         u=None):

        batch_size = target.shape[0]
        device = target.device

        jacobian, pred, rot_mat = self.forward_jacobian_and_pred_train(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist,
            rotmat_leaf=rotmat_leaf
        )

        residual = (pred - target).reshape(batch_size, 72,1)
        mse = residual.square().mean(1).squeeze()
        print(mse)
        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)
        ident = torch.eye(18, device=device).reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident
        # update = last_mse - mse
        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle-delta), mse, rot_mat


    def forward_full_withtwist(self,
                rest_J,
                v_shaped,
                transl=None,
                rot_mats=None,):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.joints_regressor_extra, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
        #
        # output = ModelOutput(
        #     vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        output = {
            'vertices': vertices,
            'joints': new_joints,
            'poses': rot_mats,
            'joints_from_verts': joints_from_verts,
        }


        return output




    def forward(self,
                pose_skeleton,
                betas,
                phis,
                global_orient,
                transl=None,
                return_verts=True,
                leaf_thetas=None):

        batch_size = pose_skeleton.shape[0]
        device = betas.device

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)
        # if leaf_thetas is not None:
        #     rest_J = vertices2joints(self.J_regressor, v_shaped)
        # else:
        #     rest_J = torch.zeros((v_shaped.shape[0], 29, 3), dtype=self.dtype, device=device)
        #     rest_J[:, :24] = vertices2joints(self.J_regressor, v_shaped)
        #
        #     leaf_number = [411, 2445, 5905, 3216, 6617]
        #     leaf_vertices = v_shaped[:, leaf_number].clone()
        #     rest_J[:, 24:] = leaf_vertices

        global_orient = self.forward_global_orient(pose_skeleton=pose_skeleton, rest_J=rest_J)
        rot_mat_twist, rotmat_leaf = self.forward_twist_and_leaf_train(rest_J=rest_J, phis=phis, global_orient=global_orient, leaf_thetas=leaf_thetas)
        v = 10

        if self.training == False:
            u = 1e-2 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)
            last_update = torch.zeros([batch_size], dtype=torch.float32, device=device)
            last_mse = torch.zeros([batch_size], dtype=torch.float32, device=device)
            params = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            for i in range(30):
                params, mse, rot_mat = self.single_iteration_train(pose_axis_angle=params, target=pose_skeleton,
                                                                           rest_J=rest_J, global_orient=global_orient,
                                                                           rot_mat_twist=rot_mat_twist,
                                                                           rotmat_leaf=rotmat_leaf, u=u)
                # print(mse)
                update = last_mse - mse

                u_index = (update > last_update) * (update > 0)
                u[u_index, :, :] /= v
                u[~u_index, :, :] *= v

                last_update = update
                last_mse = mse

            output = self.forward_full_withtwist(
                rest_J=rest_J.clone(),
                v_shaped=v_shaped.clone(),
                rot_mats=rot_mat
            )
        else:

            params0 = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            mse0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
            update0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
            u0 = 1e-2 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)

            params1, mse1, rot_mat1 = self.single_iteration_train(pose_axis_angle=params0, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u0)
            update1 = mse0 - mse1
            u_index = (update1 > update0) * (update1 > 0)
            u1 = u0.clone()
            u1[u_index, :, :] /= v
            u1[~u_index, :, :] *= v

            params2, mse2, rot_mat2 = self.single_iteration_train(pose_axis_angle=params1, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u1)
            update2 = mse1 - mse2
            u_index = (update2 > update1) * (update2 > 0)
            u2 = u1.clone()
            u2[u_index, :, :] /= v
            u2[~u_index, :, :] *= v

            params3, mse3, rot_mat3 = self.single_iteration_train(pose_axis_angle=params2, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u2)
            update3 = mse2 - mse3
            u_index = (update3 > update2) * (update3 > 0)
            u3 = u2.clone()
            u3[u_index, :, :] /= v
            u3[~u_index, :, :] *= v

            params4, mse4, rot_mat4 = self.single_iteration_train(pose_axis_angle=params3, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u3)
            update4 = mse3 - mse4
            u_index = (update4 > update3) * (update4 > 0)
            u4 = u3.clone()
            u4[u_index, :, :] /= v
            u4[~u_index, :, :] *= v

            params5, mse5, rot_mat5 = self.single_iteration_train(pose_axis_angle=params4, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.clone(), u=u4)

            output = self.forward_full_withtwist(
                rest_J=rest_J.clone(),
                v_shaped=v_shaped.clone(),
                rot_mats=rot_mat5
            )

        #
        # if leaf_thetas is not None:
        #     leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        #     leaf_thetas = quat_to_rotmat(leaf_thetas)
        #
        # batch_size = max(betas.shape[0], pose_skeleton.shape[0])
        # device = betas.device
        #
        # # 1. Add shape contribution
        # v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        #
        # # 2. Get the rest joints
        # # NxJx3 array
        # if leaf_thetas is not None:
        #     rest_J = vertices2joints(self.J_regressor, v_shaped)
        # else:
        #     rest_J = torch.zeros((v_shaped.shape[0], 29, 3),
        #                          dtype=self.dtype,
        #                          device=device)
        #     rest_J[:, :24] = vertices2joints(self.J_regressor, v_shaped)
        #
        #     leaf_number = [411, 2445, 5905, 3216, 6617]
        #     leaf_vertices = v_shaped[:, leaf_number].clone()
        #     rest_J[:, 24:] = leaf_vertices
        #
        # # 3. Get the rotation matrics
        # # rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform(
        # #     pose_skeleton,
        # #     global_orient,
        # #     phis,
        # #     rest_J.clone(),
        # #     self.children_map,
        # #     self.parents,
        # #     dtype=self.dtype,
        # #     train=self.training,
        # #     leaf_thetas=leaf_thetas)
        #
        # if self.training:
        #     rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_train(
        #         pose_skeleton, global_orient, phis,
        #         rest_J.clone(), self.children_map, self.parents, dtype=self.dtype, train=self.training,
        #         leaf_thetas=leaf_thetas)
        # else:
        #     rot_mats, rotate_rest_pose = batch_inverse_kinematics_transform_optimized(
        #         pose_skeleton, phis,
        #         rest_J.clone(), self.children_map, self.parents, dtype=self.dtype, train=self.training,
        #         leaf_thetas=leaf_thetas)
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # test_joints = True
        # if test_joints:
        #     new_joints, A = batch_rigid_transform(
        #         rot_mats,
        #         rest_J[:, :24].clone(),
        #         self.parents[:24],
        #         dtype=self.dtype)
        # else:
        #     new_joints = None
        #
        # # assert torch.mean(torch.abs(rotate_rest_pose - new_joints)) < 1e-5
        # # 4. Add pose blend shapes
        # # rot_mats: N x (J + 1) x 3 x 3
        # ident = torch.eye(3, dtype=self.dtype, device=device)
        # pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        # pose_offsets = torch.matmul(pose_feature, self.posedirs) \
        #     .view(batch_size, -1, 3)
        #
        # v_posed = pose_offsets + v_shaped
        #
        # # 5. Do skinning:
        # # W is N x V x (J + 1)
        # W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # # (N x V x (J + 1)) x (N x (J + 1) x 16)
        # num_joints = self.J_regressor.shape[0]
        # T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        #     .view(batch_size, -1, 4, 4)
        #
        # homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
        #                            dtype=self.dtype,
        #                            device=device)
        # v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        # v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        #
        # vertices = v_homo[:, :, :3, 0]
        # joints_from_verts = vertices2joints(self.joints_regressor_extra,
        #                                     vertices)
        #
        # # rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        # if transl is not None:
        #     new_joints += transl.unsqueeze(dim=1)
        #     vertices += transl.unsqueeze(dim=1)
        #     joints_from_verts += transl.unsqueeze(dim=1)
        # else:
        #     vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
        #     new_joints = new_joints - \
        #         new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        #     joints_from_verts = joints_from_verts - \
        #         joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
        #
        # output = {
        #     'vertices': vertices,
        #     'joints': new_joints,
        #     'poses': rot_mats,
        #     'joints_from_verts': joints_from_verts,
        # }
        return output



class HybrIK24OptSMPL(SMPL):
    """Extension of the SMPL for HybrIK."""

    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis',
        'left_hip',
        'right_hip',  # 2
        'spine1',
        'left_knee',
        'right_knee',  # 5
        'spine2',
        'left_ankle',
        'right_ankle',  # 8
        'spine3',
        'left_foot',
        'right_foot',  # 11
        'neck',
        'left_collar',
        'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder',
        'right_shoulder',  # 17
        'left_elbow',
        'right_elbow',  # 19
        'left_wrist',
        'right_wrist',  # 21
        'left_thumb',
        'right_thumb',  # 23
        'head',
        'left_middle',
        'right_middle',  # 26
        'left_bigtoe',
        'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]
    root_idx_17 = 0
    root_idx_smpl = 0

    def __init__(self, *args, extra_joints_regressor=None, **kwargs):
        """
        Args:
            *args: extra arguments for SMPL initialization.
            extra_joints_regressor: path to extra joint regressor. Should be
                a .npy file. If provided, extra joints are regressed and
                concatenated after the joints regressed with the official
                J_regressor or joints_regressor.
            **kwargs: extra keyword arguments for SMPL initialization.

        Returns:
            None
        """
        super(HybrIK24OptSMPL, self).__init__(
            *args,
            extra_joints_regressor=extra_joints_regressor,
            create_betas=False,
            create_global_orient=False,
            create_body_pose=False,
            create_transl=False,
            **kwargs)

        self.dtype = torch.float32
        # self.num_joints = 29
        self.num_joints = 24
        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [
            self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES
        ]
        self.SPINE3_IDX = 9
        # # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        # extend kinematic tree
        parents[:24] = self.parents
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]
        self.register_buffer('children_map',
                             self._parents_to_children(parents))
        self.register_buffer(
            'children_map_opt',
            self._parents_to_children_opt(parents))


        self.parents = parents
        self.idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [20, 21],  # 7
            [15, 22, 23, 10, 11]  # 8
            ]

        self.parent_indexs = [
            [-1],  # 0
            [-1],  # 1
            [-1],  # 2
            [-1],  # 3
            [0, 1],  # 4
            [0, 2],  # 5
            [0, 3],  # 6
            [0, 1, 4],  # 7
            [0, 2, 5],  # 8
            [0, 3, 6],  # 9
            [0, 1, 4, 7],  # 10
            [0, 2, 5, 8],  # 11
            [0, 3, 6, 9],  # 12
            [0, 3, 6, 9],  # 13
            [0, 3, 6, 9],  # 14
            [0, 3, 6, 9, 12],  # 15
            [0, 3, 6, 9, 13],  # 16
            [0, 3, 6, 9, 14],  # 17
            [0, 3, 6, 9, 13, 16],  # 18
            [0, 3, 6, 9, 14, 17],  # 19
            [0, 3, 6, 9, 13, 16, 18],  # 20
            [0, 3, 6, 9, 14, 17, 19],  # 21
            [0, 3, 6, 9, 13, 16, 18, 20],  # 22
            [0, 3, 6, 9, 14, 17, 19, 21]  # 23
            ]  # 受到影响的父节点 index

        self.idx_jacobian = [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19],
            [20, 21],
            [22, 23]
        ]  # 少了0,1,2,3

        self.index_18_to_24 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    def _parents_to_children_opt(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        # children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[-1] = -1
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children


    def forward_global_orient(self,
                pose_skeleton,
                rest_J):


        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()  # 防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # TODO
        if self.training:
            global_orient_mat = batch_get_pelvis_orient_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)
        else:
            global_orient_mat = batch_get_pelvis_orient_svd_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)

        return global_orient_mat


    def forward_twist_and_leaf_train(self,
                rest_J,
                phis,
                global_orient,
                leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        rot_mat_local = [global_orient]
        # leaf nodes rot_mats
        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            rotmat_leaf = quat_to_rotmat(leaf_thetas)
            rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])
        else:
            rotmat_leaf_ = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(batch_size, 5, 1, 1)
            # rotmat_leaf_ = None


        for indices in range(1, 24):

            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
            ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)

            # (B, K, 1, 1)
            cos, sin = torch.split(phis[:, indices - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)

            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat_local.append(rot_mat_spin)

        # (B, K + 1, 3, 3)
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return rot_mats, rotmat_leaf_

    def forward_jacobian_and_pred_train(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw, rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2].clone(), R_derivative[:, parents_1, parents_1].clone())
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2].clone()
                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @  DR_k_1_k_1
                    # rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone().matmul(DR_k_1_k_1)



                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......


                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_2].clone()   ##### BP bug

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]])
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3].clone()
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @ DR_k_1_k_2
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]].clone() @ DR_k_2_k_2 @ rot_mat_local[parent_index[-2]].clone()

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_3].clone()

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]])
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4].clone()
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4].clone()
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @ DR_k_1_k_3
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]].clone() @ DR_k_2_k_3 @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]].clone() @ DR_k_3_k_3 @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()


                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_4].clone()

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]])

                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5].clone()
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5].clone()
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5].clone()
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @ DR_k_1_k_4
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]].clone() @ DR_k_2_k_4 @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]].clone() @ DR_k_3_k_4 @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]].clone() @ DR_k_4_k_4 @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_5].clone()


                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]])

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6].clone()
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6].clone()
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6].clone()
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6].clone()
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6].clone()

                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @ DR_k_1_k_5
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]].clone() @ DR_k_2_k_5 @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]].clone() @ DR_k_3_k_5 @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]].clone() @ DR_k_4_k_5 @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR5 = rot_mat_chain[parent_index[-7]].clone() @ DR_k_5_k_5 @ rot_mat_local[parent_index[-5]].clone() @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_6].clone()

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR5, rot_mat_local[parent_index[-1]])


                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7].clone()
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7].clone()
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7].clone()
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7].clone()
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7].clone()
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7].clone()
                    rot_mat_withDR1 = rot_mat_chain[parent_index[-3]].clone() @ DR_k_1_k_6
                    rot_mat_withDR2 = rot_mat_chain[parent_index[-4]].clone() @ DR_k_2_k_6 @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR3 = rot_mat_chain[parent_index[-5]].clone() @ DR_k_3_k_6 @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR4 = rot_mat_chain[parent_index[-6]].clone() @ DR_k_4_k_6 @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR5 = rot_mat_chain[parent_index[-7]].clone() @ DR_k_5_k_6 @ rot_mat_local[parent_index[-5]].clone() @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()
                    rot_mat_withDR6 = rot_mat_chain[parent_index[-8]].clone() @ DR_k_6_k_6 @ rot_mat_local[parent_index[-6]].clone() @ rot_mat_local[parent_index[-5]].clone() @ rot_mat_local[parent_index[-4]].clone() @ rot_mat_local[parent_index[-3]].clone() @ rot_mat_local[parent_index[-2]].clone()

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_withDR0 = rot_mat_chain[parent_index[-2]].clone() @ R_derivative[:, parents_1, parents_7].clone()

                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + \
                             torch.matmul(rot_mat_withDR1, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR2, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR3, rot_mat_local[parent_index[-1]]) +  \
                             torch.matmul(rot_mat_withDR4, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR5, rot_mat_local[parent_index[-1]]) + \
                             torch.matmul(rot_mat_withDR6, rot_mat_local[parent_index[-1]])

                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats



    def forward_jacobian_and_pred(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # rot_mat_chain[:, parents_1] = torch.matmul(
            #     rot_mat_chain[:, parents_2],
            #     rot_mat)
            # rot_mat_local[:, parents_1] = rot_mat

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2].clone(), R_derivative[:, parents_1, parents_1].clone())
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    # rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    # rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    # rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                    # rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                    # for index_r in range(1, tree_len - 1):
                    #     rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                    #                                    rot_mat_local_withDR1[:, index_r])

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_1.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_2]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_2].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_2.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), DR_k_2_k_2.clone())
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_3].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_3.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), DR_k_2_k_3.clone())
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), DR_k_3_k_3.clone())
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_4]
                    #
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_4].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_4.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), DR_k_2_k_4.clone())
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), DR_k_3_k_4.clone())
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), DR_k_4_k_4.clone())
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_5]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_5].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())

                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_5.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), DR_k_2_k_5.clone())
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), DR_k_3_k_5.clone())
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), DR_k_4_k_5.clone())
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())

                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), DR_k_5_k_5.clone())
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_6]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_6].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), rot_mat_local[index_r].clone())

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    rot_mat_withDR6 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), DR_k_1_k_6.clone())
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), DR_k_2_k_6.clone())
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r].clone())
                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), DR_k_3_k_6.clone())
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())
                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), DR_k_4_k_6.clone())
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())
                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), DR_k_5_k_6.clone())
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), rot_mat_local[index_r].clone())
                        if index_r == parent_index[-7]:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6.clone(), DR_k_6_k_6.clone())
                        else:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6.clone(), rot_mat_local[index_r].clone())

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), R_derivative[:, parents_1, parents_7].clone())
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0.clone(), rot_mat_local[index_r].clone())

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2.clone(), rot_mat_local[index_r]).clone()
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5.clone(), rot_mat_local[index_r].clone())
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6.clone(), rot_mat_local[index_r].clone())
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats

    def forward_jacobian_and_pred_v2(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # rot_mat_chain[:, parents_1] = torch.matmul(
            #     rot_mat_chain[:, parents_2],
            #     rot_mat)
            # rot_mat_local[:, parents_1] = rot_mat

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )
                # rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                # leaf_cnt += 1
                # # rot_mat_chain[:, indices] = torch.matmul(
                # #     rot_mat_chain[:, parents_1],
                # #     rot_mat)
                # # rot_mat_local[:, indices] = rot_mat
                # rot_mat_chain.append(torch.matmul(
                #     rot_mat_chain[parents_1],
                #     rot_mat))
                # rot_mat_local.append(rot_mat)

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    # rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    # rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    # rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                    # rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                    # for index_r in range(1, tree_len - 1):
                    #     rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                    #                                    rot_mat_local_withDR1[:, index_r])

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_1)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_2]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_2])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_2)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_2)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_3])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_3)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_3)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_3)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_4]
                    #
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_4])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_4)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_4)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_4)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_4)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_5]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_5])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_5)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_5)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_5)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_5)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_5)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_6]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_6])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    rot_mat_withDR6 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_6)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_6)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_6)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_6)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_6)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                        if index_r == parent_index[-7]:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, DR_k_6_k_6)
                        else:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_7])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats

    def single_iteration_train(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         global_orient,
                         rot_mat_twist=None,
                         rotmat_leaf=None,
                         u=None):

        batch_size = target.shape[0]
        device = target.device

        if self.training == True:
            # jacobian, pred, rot_mat = self.forward_jacobian_and_pred_train(
            #     pose_axis_angle=pose_axis_angle,
            #     pose_skeleton=target,
            #     rest_J=rest_J,
            #     global_orient=global_orient,
            #     rot_mat_twist=rot_mat_twist,
            #     rotmat_leaf=rotmat_leaf
            # )
            # jacobian, pred, rot_mat = self.forward_jacobian_and_pred(
            #     pose_axis_angle=pose_axis_angle,
            #     pose_skeleton=target,
            #     rest_J=rest_J,
            #     global_orient=global_orient,
            #     rot_mat_twist=rot_mat_twist,
            #     rotmat_leaf=rotmat_leaf
            # )
            jacobian, pred, rot_mat = self.forward_jacobian_and_pred_v2(
                pose_axis_angle=pose_axis_angle,
                pose_skeleton=target,
                rest_J=rest_J,
                global_orient=global_orient,
                rot_mat_twist=rot_mat_twist,
                rotmat_leaf=rotmat_leaf
            )
        else:
            jacobian, pred, rot_mat = self.forward_jacobian_and_pred_train(
                pose_axis_angle=pose_axis_angle,
                pose_skeleton=target,
                rest_J=rest_J,
                global_orient=global_orient,
                rot_mat_twist=rot_mat_twist,
                rotmat_leaf=rotmat_leaf
            )


        residual = (pred - target).reshape(batch_size, 72,1)
        mse = residual.square().mean(1).squeeze()
        # print(mse)
        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)
        ident = torch.eye(18, device=device).reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident
        # update = last_mse - mse
        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle-delta), mse, rot_mat


    def forward_full_withtwist(self,
                rest_J,
                v_shaped,
                transl=None,
                rot_mats=None,):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.joints_regressor_extra, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
        #
        # output = ModelOutput(
        #     vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        output = {
            'vertices': vertices,
            'joints': new_joints,
            'poses': rot_mats,
            'joints_from_verts': joints_from_verts,
        }


        return output




    def forward(self,
                pose_skeleton,
                betas,
                phis,
                global_orient,
                transl=None,
                return_verts=True,
                leaf_thetas=None):

        batch_size = pose_skeleton.shape[0]
        device = betas.device

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        global_orient = self.forward_global_orient(pose_skeleton=pose_skeleton, rest_J=rest_J)
        rot_mat_twist, rotmat_leaf = self.forward_twist_and_leaf_train(rest_J=rest_J, phis=phis, global_orient=global_orient, leaf_thetas=leaf_thetas)
        v = 10

        if self.training == False:
            u = 1e-2 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)
            last_update = torch.zeros([batch_size], dtype=torch.float32, device=device)
            last_mse = torch.zeros([batch_size], dtype=torch.float32, device=device)
            params = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            for i in range(30):
                params, mse, rot_mat = self.single_iteration_train(pose_axis_angle=params, target=pose_skeleton,
                                                                           rest_J=rest_J, global_orient=global_orient,
                                                                           rot_mat_twist=rot_mat_twist,
                                                                           rotmat_leaf=rotmat_leaf, u=u)
                # print(mse)
                update = last_mse - mse

                u_index = (update > last_update) * (update > 0)
                u[u_index, :, :] /= v
                u[~u_index, :, :] *= v

                last_update = update
                last_mse = mse

            output = self.forward_full_withtwist(
                rest_J=rest_J.clone(),
                v_shaped=v_shaped.clone(),
                rot_mats=rot_mat
            )
        else:

            params0 = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
            mse0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
            update0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
            u0 = 1e-2 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)

            params1, mse1, rot_mat1 = self.single_iteration_train(pose_axis_angle=params0, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u0)
            update1 = mse0 - mse1
            u_index = (update1 > update0) * (update1 > 0)
            u1 = u0.clone()
            u1[u_index, :, :] /= v
            u1[~u_index, :, :] *= v

            params2, mse2, rot_mat2 = self.single_iteration_train(pose_axis_angle=params1, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u1)
            update2 = mse1 - mse2
            u_index = (update2 > update1) * (update2 > 0)
            u2 = u1.clone()
            u2[u_index, :, :] /= v
            u2[~u_index, :, :] *= v

            params3, mse3, rot_mat3 = self.single_iteration_train(pose_axis_angle=params2, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u2)
            update3 = mse2 - mse3
            u_index = (update3 > update2) * (update3 > 0)
            u3 = u2.clone()
            u3[u_index, :, :] /= v
            u3[~u_index, :, :] *= v

            params4, mse4, rot_mat4 = self.single_iteration_train(pose_axis_angle=params3, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.detach().clone(), u=u3)
            update4 = mse3 - mse4
            u_index = (update4 > update3) * (update4 > 0)
            u4 = u3.clone()
            u4[u_index, :, :] /= v
            u4[~u_index, :, :] *= v

            params5, mse5, rot_mat5 = self.single_iteration_train(pose_axis_angle=params4, target=pose_skeleton,
                                                                     rest_J=rest_J.clone(),
                                                                     global_orient=global_orient.clone(),
                                                                     rot_mat_twist=rot_mat_twist.clone(),
                                                                     rotmat_leaf=rotmat_leaf.clone(), u=u4)

            output = self.forward_full_withtwist(
                rest_J=rest_J.clone(),
                v_shaped=v_shaped.clone(),
                rot_mats=rot_mat5
            )

        return output


class HybrIKOptSMPLV2(nn.Module):
    NUM_JOINTS = 23
    NUM_BODY_JOINTS = 23
    NUM_BETAS = 10
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip',  # 2
        'spine1', 'left_knee', 'right_knee',  # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',  # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',  # 15
        'left_shoulder', 'right_shoulder',  # 17
        'left_elbow', 'right_elbow',  # 19
        'left_wrist', 'right_wrist',  # 21
        'left_thumb', 'right_thumb',  # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'  # 28
    ]
    LEAF_NAMES = [
        'head', 'left_middle', 'right_middle', 'left_bigtoe', 'right_bigtoe'
    ]

    # leaf 15, 22, 23, 10, 11
    root_idx_17 = 0
    root_idx_smpl = 0

    # extra_joints_regressor = extra_joints_regressor,
    # create_betas = False,
    # create_global_orient = False,
    # create_body_pose = False,
    # create_transl = False,

    def __init__(self,
                 model_path,
                 h36m_jregressor,
                 gender='neutral',
                 dtype=torch.float32,
                 num_joints=24):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(HybrIKOptSMPLV2, self).__init__()

        self.ROOT_IDX = self.JOINT_NAMES.index('pelvis')
        self.LEAF_IDX = [self.JOINT_NAMES.index(name) for name in self.LEAF_NAMES]
        self.SPINE3_IDX = 9

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender

        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        # Vertices to Human3.6M Joints location (17, 6890)
        self.register_buffer(
            'J_regressor_h36m',
            to_tensor(to_np(h36m_jregressor), dtype=dtype))

        self.num_joints = num_joints

        # indices of parents for each joints
        parents = torch.zeros(len(self.JOINT_NAMES), dtype=torch.long)
        parents[:(self.NUM_JOINTS + 1)] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        # extend kinematic tree
        parents[24] = 15
        parents[25] = 22
        parents[26] = 23
        parents[27] = 10
        parents[28] = 11
        if parents.shape[0] > self.num_joints:
            parents = parents[:24]

        self.register_buffer(
            'children_map',
            self._parents_to_children(parents))

        self.register_buffer(
            'children_map_opt',
            self._parents_to_children_opt(parents))

        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

        self.idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [20, 21],  # 7
            [15, 22, 23, 10, 11]  # 8
        ]

        self.parent_indexs = [
            [-1],  # 0
            [-1],  # 1
            [-1],  # 2
            [-1],  # 3
            [0, 1],  # 4
            [0, 2],  # 5
            [0, 3],  # 6
            [0, 1, 4],  # 7
            [0, 2, 5],  # 8
            [0, 3, 6],  # 9
            [0, 1, 4, 7],  # 10
            [0, 2, 5, 8],  # 11
            [0, 3, 6, 9],  # 12
            [0, 3, 6, 9],  # 13
            [0, 3, 6, 9],  # 14
            [0, 3, 6, 9, 12],  # 15
            [0, 3, 6, 9, 13],  # 16
            [0, 3, 6, 9, 14],  # 17
            [0, 3, 6, 9, 13, 16],  # 18
            [0, 3, 6, 9, 14, 17],  # 19
            [0, 3, 6, 9, 13, 16, 18],  # 20
            [0, 3, 6, 9, 14, 17, 19],  # 21
            [0, 3, 6, 9, 13, 16, 18, 20],  # 22
            [0, 3, 6, 9, 14, 17, 19, 21]  # 23
        ]  # 受到影响的父节点 index

        self.idx_jacobian = [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19],
            [20, 21],
            [22, 23]
        ]  # 少了0,1,2,3

        self.index_18_to_24 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    def _parents_to_children(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def _parents_to_children_opt(self, parents):
        children = torch.ones_like(parents) * -1
        for i in range(self.num_joints):
            if children[parents[i]] < 0:
                children[parents[i]] = i
        for i in self.LEAF_IDX:
            if i < children.shape[0]:
                children[i] = -1

        # children[self.SPINE3_IDX] = -3
        children[0] = 3
        children[-1] = -1
        children[self.SPINE3_IDX] = self.JOINT_NAMES.index('neck')

        return children

    def forward(self,
                pose_axis_angle,
                betas,
                global_orient,
                transl=None,
                return_verts=True):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]

        # concate root orientation with thetas
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # Translate thetas to rotation matrics
        pose2rot = True
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m = lbs(betas, full_pose, self.v_template,
                                                                 self.shapedirs, self.posedirs,
                                                                 self.J_regressor, self.J_regressor_h36m, self.parents,
                                                                 self.lbs_weights, pose2rot=pose2rot, dtype=self.dtype)

        if transl is not None:
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(
                1).detach()

        output = ModelOutput(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output

    def forward_generate_dataset(self,
                                 pose_axis_angle,
                                 betas,
                                 global_orient,
                                 transl=None,
                                 return_verts=True):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]

        # concate root orientation with thetas
        if global_orient is not None:
            full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        else:
            full_pose = pose_axis_angle

        # Translate thetas to rotation matrics
        pose2rot = True
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints, rot_mats, joints_from_verts_h36m, twist_angle = lbs_generation(betas, full_pose,
                                                                                         self.v_template,
                                                                                         self.shapedirs, self.posedirs,
                                                                                         self.J_regressor,
                                                                                         self.J_regressor_h36m,
                                                                                         self.parents,
                                                                                         self.lbs_weights,
                                                                                         pose2rot=pose2rot,
                                                                                         dtype=self.dtype)

        # verts, J_transformed, rot_mats, J_from_verts, twist_angle
        if transl is not None:
            # apply translations
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            # joints_from_verts_h36m += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(1).detach()
            joints = joints - joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts_h36m = joints_from_verts_h36m - joints_from_verts_h36m[:, self.root_idx_17, :].unsqueeze(
                1).detach()

        output = ModelOutput(
            vertices=vertices, joints=joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts_h36m)
        return output, twist_angle

    def hybrik(self,
               pose_skeleton,
               betas,
               phis,
               global_orient,
               transl=None,
               return_verts=True,
               leaf_thetas=None):
        ''' Inverse pass for the SMPL model

            Parameters
            ----------
            pose_skeleton: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint locations in
                (X, Y, Z) format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_skeleton.shape[0]

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            leaf_thetas = quat_to_rotmat(leaf_thetas)

        vertices, new_joints, rot_mats, joints_from_verts = hybrik(
            betas, global_orient, pose_skeleton, phis,
            self.v_template, self.shapedirs, self.posedirs,
            self.J_regressor, self.J_regressor_h36m, self.parents, self.children_map,
            self.lbs_weights, dtype=self.dtype, train=self.training,
            leaf_thetas=leaf_thetas)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            new_joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = new_joints - new_joints[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_global_orient(self,
                              pose_skeleton,
                              rest_J):

        # batch_size = pose_skeleton.shape[0]

        #  joints number we need is 24 - 5 - 1    (five leaf_thetas and one root rotation)

        # v_shaped =  self.v_template + blend_shapes(betas, self.shapedirs)
        # rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()  # 防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # TODO
        if self.training:
            global_orient_mat = batch_get_pelvis_orient_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)
        else:
            global_orient_mat = batch_get_pelvis_orient_svd_v2(
                rel_pose_skeleton.clone(), rel_rest_pose.clone(), self.parents, self.children_map, self.dtype)

        return global_orient_mat

    def forward_Opt(self,
                    pose_axis_angle,
                    pose_skeleton,
                    betas,
                    phis,
                    global_orient,
                    transl=None,
                    return_verts=True,
                    leaf_thetas=None):
        ''' Forward pass for the SMPL model

            With Twist input and leaf_thetas !!!!!!!

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)

            Returns
            -------
        '''
        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        # # concate root orientation with thetas
        # if global_orient is not None:
        #     full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        # else:
        #     full_pose = pose_axis_angle

        #  joints number we need is 24 - 5 - 1    (five leaf_thetas and one root rotation)

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        # rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parents[1:]].clone()
        # rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
        #
        # the predicted final pose
        # final_pose_skeleton = torch.unsqueeze(rotate_rest_pose.clone(), dim=-1)
        final_pose_skeleton = rotate_rest_pose.clone()
        final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, [0]] + rel_rest_pose[:, [0]]

        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        leaf_thetas = quat_to_rotmat(leaf_thetas)
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 15, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [10, 11, 20, 21],  # 7
            [22, 23],  # 8
            [24, 25, 26, 27, 28]  # 9
        ]

        idx_levs = idx_levs[:-1]

        for idx_lev in range(1, len(idx_levs)):
            indices = idx_levs[idx_lev]
            if idx_lev == len(idx_levs) - 1:
                # leaf nodes
                rot_mat = leaf_rot_mats[:, :, :, :]
                parent_indices = [15, 22, 23, 10, 11]
                rot_mat_local[:, parent_indices] = rot_mat
                if (torch.det(rot_mat) < 0).any():
                    print('Something wrong.')

            elif idx_lev == 3:
                # three children
                idx = indices[0]
                self.children_map[indices] = 12

                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map[indices]] - rotate_rest_pose[:, indices]

                # orig_vec = rel_pose_skeleton[:, self.children_map[indices]]
                # template_vec = rel_rest_pose[:, self.children_map[indices]]

                # norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                # orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                # diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                # big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                # orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )

                child_rest_loc = rel_rest_pose[:, self.children_map[indices]]  # need rotation back ?

                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
                # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices[0]] = rot_mat

            else:
                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map[indices]] - rotate_rest_pose[:, indices]

                # orig_vec = rel_pose_skeleton[:, self.children_map[indices]]
                # template_vec = rel_rest_pose[:, self.children_map[indices]]

                # norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                # orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                # diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                # big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                # orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )

                child_rest_loc = rel_rest_pose[:, self.children_map[indices]]  # need rotation back ?

                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
                # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

        # (B, K + 1, 3, 3)
        # rot_mats = torch.stack(rot_mat_local, dim=1)
        rot_mats = rot_mat_local

        test_joints = True
        if test_joints:
            J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24],
                                                     dtype=self.dtype)
        else:
            J_transformed = None

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_rest_J(self, betas):

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        return rest_J, v_shaped

    def forward_light(self,
                      pose_axis_angle,
                      pose_skeleton,
                      rest_J,
                      phis,
                      global_orient,
                      transl=None,
                      return_verts=True,
                      leaf_thetas=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        # v_shaped =  self.v_template + blend_shapes(betas, self.shapedirs)
        # rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
        #
        final_pose_skeleton = rotate_rest_pose.clone()
        final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, [0]] + rel_rest_pose[:, [0]]

        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        leaf_thetas = quat_to_rotmat(leaf_thetas)
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 15, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [10, 11, 20, 21],  # 7
            [22, 23],  # 8
            [24, 25, 26, 27, 28]  # 9
        ]

        idx_levs = idx_levs[:-1]

        import time
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for idx_lev in range(1, len(idx_levs)):
            indices = idx_levs[idx_lev]
            if idx_lev == len(idx_levs) - 1:
                # leaf nodes
                # rot_mat = leaf_rot_mats[:, :, :, :]
                # parent_indices = [15, 22, 23, 10, 11]
                # rot_mat_local[:, parent_indices] = rot_mat
                # if (torch.det(rot_mat) < 0).any():
                #     print('Something wrong.')
                rot_mat = leaf_rot_mats[:, :, :, :]
                parent_indices = [15, 22, 23, 10, 11]
                rot_mat_local[:, indices] = rot_mat[:, 1:3]
                if (torch.det(rot_mat) < 0).any():
                    print('Something wrong.')
            elif idx_lev == 3:
                # three children
                idx = indices[0]
                # self.children_map_opt[indices] = 12

                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - rotate_rest_pose[:, indices]

                orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
                template_vec = rel_rest_pose[:, self.children_map_opt[indices]]

                norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )

                child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?

                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices[0]] = rot_mat
            else:
                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - rotate_rest_pose[:, indices]

                orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
                template_vec = rel_rest_pose[:, self.children_map_opt[indices]]

                norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )
                child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
                # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        print(elapsed)

        # (B, K + 1, 3, 3)
        rot_mats = rot_mat_local

        test_joints = True
        if test_joints:
            # import time
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24],
                                                     dtype=self.dtype)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed)
        else:
            J_transformed = None

        new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return new_joints

    def forward_light_V2(self,
                         pose_axis_angle,
                         pose_skeleton,
                         rest_J,
                         phis,
                         global_orient,
                         transl=None,
                         return_verts=True,
                         leaf_thetas=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        # v_shaped =  self.v_template + blend_shapes(betas, self.shapedirs)
        # rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
        # #
        # final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        # final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, [0]] + rel_rest_pose[:, [0]]

        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        leaf_thetas = quat_to_rotmat(leaf_thetas)
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 15, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [10, 11, 20, 21],  # 7
            [22, 23],  # 8
            [24, 25, 26, 27, 28]  # 9
        ]

        idx_levs = idx_levs[:-1]

        # import time
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()

        for idx_lev in range(1, len(idx_levs) - 1):
            indices = idx_levs[idx_lev]
            # if idx_lev == len(idx_levs) - 1:
            #
            #     rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            #         rot_mat_chain[:, self.parents[indices]],
            #         rel_rest_pose[:, indices]
            #     )
            #     rot_mat = leaf_rot_mats[:, :, :, :]
            #     # parent_indices = [15, 22, 23, 10, 11]
            #     rot_mat_local[:, indices] = rot_mat[:,1:3]
            #
            #
            #     if (torch.det(rot_mat) < 0).any():
            #         print('Something wrong.')
            # # elif idx_lev == 3:
            # #     # three children
            # #     idx = indices[0]
            # #     # self.children_map_opt[indices] = 12
            # #
            # #     len_indices = len(indices)
            # #     # (B, K, 3, 1)
            # #     rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            # #         rot_mat_chain[:, self.parents[indices]],
            # #         rel_rest_pose[:, indices]
            # #     )
            # #
            # #     orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            # #     orig_vec = torch.matmul(
            # #         rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
            # #         orig_vec
            # #     )
            # #
            # #     child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # #     # (B, K, 1, 1)
            # #     # child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            # #     child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)
            # #
            # #     # (B, K, 3, 1)
            # #     axis = torch.cross(child_rest_loc, orig_vec, dim=2)
            # #     axis_norm = torch.norm(axis, dim=2, keepdim=True)
            # #     # # (B, K, 3, 1)
            # #     # rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            # #     #     rot_mat_chain[:, self.parents[indices]],
            # #     #     rel_rest_pose[:, indices]
            # #     # )
            # #     # # (B, 3, 1)
            # #     # # child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - rotate_rest_pose[:, indices]
            # #     # child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - final_pose_skeleton[:, indices]
            # #     #
            # #     # orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            # #     # template_vec = rel_rest_pose[:, self.children_map_opt[indices]]
            # #     #
            # #     # norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1
            # #     #
            # #     # orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3
            # #     #
            # #     # diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
            # #     # big_diff_idx = torch.where(diff > 15 / 1000)[0]
            # #     #
            # #     # # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
            # #     # child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
            # #     # orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
            # #     # # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
            # #     # child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)
            # #     #
            # #     # child_final_loc = torch.matmul(
            # #     #     rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
            # #     #     child_final_loc
            # #     # )
            # #     #
            # #     # child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # #     #
            # #     # # (B, K, 1, 1)
            # #     # child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            # #     # child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)
            # #     #
            # #     # # (B, K, 3, 1)
            # #     # axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
            # #     # axis_norm = torch.norm(axis, dim=2, keepdim=True)
            # #
            # #     # (B, K, 1, 1)
            # #     cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # #     sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # #
            # #     # (B, K, 3, 1)
            # #     axis = axis / (axis_norm + 1e-8)
            # #
            # #
            # #     # Convert location revolve to rot_mat by rodrigues
            # #     # (B, K, 1, 1)
            # #     rx, ry, rz = torch.split(axis, 1, dim=2)
            # #     zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            # #
            # #     K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
            # #         .view((batch_size, len_indices, 3, 3))
            # #     ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # #     rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # #
            # #     # Convert spin to rot_mat
            # #     # (B, K, 3, 1)
            # #     spin_axis = child_rest_loc / child_rest_norm
            # #     # (B, K, 1, 1)
            # #     rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            # #     zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            # #     K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
            # #         .view((batch_size, len_indices, 3, 3))
            # #     ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # #     # (B, K, 1, 1)
            # #     phi_indices = [item - 1 for item in indices]
            # #     cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            # #     cos = torch.unsqueeze(cos, dim=3)
            # #     sin = torch.unsqueeze(sin, dim=3)
            # #     rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # #     rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # #
            # #     if (torch.det(rot_mat) < 0).any():
            # #         print(
            # #             2,
            # #             torch.det(rot_mat_loc) < 0,
            # #             torch.det(rot_mat_spin) < 0
            # #         )
            # #
            # #     rot_mat_chain[:, indices] = torch.matmul(
            # #         rot_mat_chain[:, self.parents[indices]],
            # #         rot_mat)
            # #     rot_mat_local[:, indices[0]] = rot_mat
            #
            # else:
            len_indices = len(indices)
            # (B, K, 3, 1)
            rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rel_rest_pose[:, indices]
            )

            orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            if idx_lev == 3:
                orig_vec = 0 * orig_vec

            orig_vec = torch.matmul(
                rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                orig_vec
            )

            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            # child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = torch.cross(child_rest_loc, orig_vec, dim=2)
            axis_norm = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 1, 1)
            # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
            # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
            cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

            # (B, K, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]
            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            if (torch.det(rot_mat) < 0).any():
                print(
                    2,
                    torch.det(rot_mat_loc) < 0,
                    torch.det(rot_mat_spin) < 0
                )

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rot_mat)
            rot_mat_local[:, indices] = rot_mat

        indices = idx_levs[-1]
        rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            rot_mat_chain[:, self.parents[indices]],
            rel_rest_pose[:, indices]
        )

        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # print(elapsed)

        # (B, K + 1, 3, 3)
        rot_mats = rot_mat_local

        # test_joints = True
        # if test_joints:
        #     # import time
        #     # torch.cuda.synchronize()
        #     # start_time = time.perf_counter()
        #     J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)
        #     # torch.cuda.synchronize()
        #     # elapsed = time.perf_counter() - start_time
        #     # print(elapsed)
        # else:
        #     J_transformed = None

        # new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return new_joints

    def forward_light_V2_withtwist(self,
                                   pose_axis_angle,
                                   pose_skeleton,
                                   rest_J,
                                   global_orient,
                                   rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()   ### 修改防止梯度截断
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        for idx_lev in range(1, len(self.idx_levs) - 1):
            indices = self.idx_levs[idx_lev]

            len_indices = len(indices)
            # (B, K, 3, 1)
            rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rel_rest_pose[:, indices]
            )

            orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            if idx_lev == 3:
                orig_vec = 0 * orig_vec

            orig_vec = torch.matmul(
                rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                orig_vec
            )
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?

            # (B, K, 3, 1)
            axis = torch.cross(child_rest_loc, orig_vec, dim=2)
            axis_norm = torch.norm(axis, dim=2, keepdim=True)
            # (B, K, 1, 1)
            # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
            # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
            cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = axis / (axis_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            rot_mat_spin = rot_mat_twist[:, indices]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rot_mat)
            rot_mat_local[:, indices] = rot_mat

        indices = self.idx_levs[-1]
        rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            rot_mat_chain[:, self.parents[indices]],
            rel_rest_pose[:, indices]
        )

        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return new_joints

    def forward_jacobian(self,
                         pose_axis_angle,
                         pose_skeleton,
                         rest_J,
                         phis,
                         global_orient,
                         transl=None,
                         leaf_thetas=None,
                         rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3).repeat(batch_size, 24, 1, 1)
        rot_mat_local = rot_mat_chain.clone()
        # rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        # rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 72, 18], dtype=torch.float32, device=device)
        # R_derivative = torch.eye(3).cuda().reshape(1,1, 3, 3).repeat(24, 18, 1, 1)
        R_derivative = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3).repeat(24, 24, 1, 1)
        # R_derivative = torch.zeros((24, 18, 3, 3), dtype=torch.float32, device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        q_derivative = torch.zeros((24, 18, 3, 1), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1

        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)

        # self.idx_levs = [
        #     [0],  # 0
        #     [3],  # 1
        #     [6],  # 2
        #     [9],  # 3
        #     [1, 2, 12, 13, 14],  # 4
        #     [4, 5, 15, 16, 17],  # 5
        #     [7, 8, 18, 19],  # 6
        #     [10, 11, 20, 21],  # 7
        #     [22, 23]  # 8
        #     ]

        # idx_trees = [
        #     [3],  # 1
        #     [6],  # 2
        #     [9],  # 3
        #     [1, 2, 12, 13, 14],  # 4
        #     [4, 5, 16, 17],  # 5
        #     [7, 8, 18, 19],  # 6
        #     [20, 21],  # 7
        #     ]

        # idx_trees = [
        #     [1, 2, 3],  # 2
        #     [4, 5, 6],  # 5
        #     [7, 8, 9],  # 8
        #     [12, 13, 14],  # 11
        #     [16, 17],  # 13
        #     [18, 19],  # 15
        #     [20, 21],  # 17
        #     ]
        #
        #
        # children_num = [23, 3, 3, 14, 3, 3, 13, 1, 1, 12, 1, 4, 4, 3, 3, 2, 2, 1, 1]  #受到影响的子节点的数量
        # parent_num = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8]  # 受到影响的父节点的数量
        #
        # index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, 12, -1, 13, 14, 15, 16, 17, -1,
                          -1]  # 受到影响的父节点的数量

        parent_indexs = [
            [-1],  # 1
            [1],  # 1
            [2],  # 2
            [3],  # 3
            [1, 4],  # 4
            [2, 5],  # 5
            [3, 6],  # 6
            [1, 4, 7],  # 7
            [2, 5, 8],  # 8
            [3, 6, 9],  # 9
            [1, 4, 7, 10],  # 10
            [2, 5, 8, 11],  # 11
            [3, 6, 9, 12],  # 12
            [3, 6, 9, 13],  # 13
            [3, 6, 9, 14],  # 14
            [3, 6, 9, 12, 15],  # 15
            [3, 6, 9, 13, 16],  # 16
            [3, 6, 9, 14, 17],  # 17
            [3, 6, 9, 13, 16, 18],  # 18
            [3, 6, 9, 14, 17, 19],  # 19
            [3, 6, 9, 13, 16, 18, 20],  # 20
            [3, 6, 9, 14, 17, 19, 21],  # 21
            [3, 6, 9, 13, 16, 18, 20, 22],  # 22
            [3, 6, 9, 14, 17, 19, 21, 23],  # 23
        ]  # 受到影响的父节点 index

        # parent_indexs = [
        #     [-1],  # 1
        #     [-1],  # 1
        #     [-1],  # 2
        #     [-1],  # 3
        #     [1],  # 4
        #     [2],  # 5
        #     [3],  # 6
        #     [1, 4],  # 7
        #     [2, 5],  # 8
        #     [3, 6],  # 9
        #     [1, 4, 7],  # 10
        #     [2, 5, 8],  # 11
        #     [3, 6, 9],  # 12
        #     [3, 6, 9],  # 13
        #     [3, 6, 9,],  # 14
        #     [3, 6, 9, 12],  # 15
        #     [3, 6, 9, 13],  # 16
        #     [3, 6, 9, 14],  # 17
        #     [3, 6, 9, 13, 16],  # 18
        #     [3, 6, 9, 14, 17],  # 19
        #     [3, 6, 9, 13, 16, 18],  # 20
        #     [3, 6, 9, 14, 17, 19],  # 21
        #     [3, 6, 9, 13, 16, 18, 20],  # 22
        #     [3, 6, 9, 14, 17, 19, 21]  # 23
        #     ]  # 受到影响的父节点 index

        tree_sub = [  # 把人体的树分成5条树
            [1, 4, 7, 10],  # 1
            [2, 5, 8, 11],  # 2
            [3, 6, 9, 14, 17, 19, 21, 23],  # 3
            [3, 6, 9, 12, 15],  # 4
            [3, 6, 9, 13, 16, 18, 20, 22]  # 5
        ]
        #
        # idx_jacobian = [
        #     [3],  # 1
        #     [6],  # 2
        #     [9],  # 3
        #     [1], [2], [12], [13], [14],  # 4
        #     [4], [5], [16], [17],  # 5
        #     [7], [8], [18], [19],  # 6
        #     [20], [21],  # 7
        #     ]  #少了 0,10,11,15,22,23

        # idx_jacobian = [
        #     [3],  # 1
        #     [1], [2], [12], [13], [14],  # 4
        #     [4], [5], [16], [17],  # 5
        #     [7], [8], [18], [19],  # 6
        #     [20], [21],  # 7
        #     ]  #少了 0,10,11,15,22,23
        # idx_jacobian = [
        #     [1], [2],  # 4
        #     [4], [5],  # 5
        #     [7], [8],  # 6
        #     ]  #少了 0,10,11,15,22,23
        #  [16], [17], [18], [19], [20], [21]
        idx_jacobian = [
            [1], [2],  # 4
            [4], [5],  # 5
            [7], [8],  # 6
            [3], [6], [9], [12],
            [13], [14], [16], [17], [19]]  # 少了 0,10,11,15,22,23

        ## 第九个节点有三个字节点，所以index 9 的时候如果计算三次

        ## 我应该按着树状图的节点的输出循环开始往下写，每个节点输出都有对应的影响到的输入x，这样代码效率最高

        for idx_lev in range(len(idx_jacobian)):

            indices = idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            parents_8 = self.parents[parents_7]

            orig_vec_unrotate = rel_pose_skeleton[:, self.children_map_opt[indices]]
            if idx_lev == 8:
                orig_vec_unrotate = 0 * orig_vec_unrotate

            orig_vec = torch.matmul(
                rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?

            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=2)
            w_norm = torch.norm(w, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            rot_mat_spin = rot_mat_twist[:, indices]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rot_mat)
            rot_mat_local[:, indices] = rot_mat

            q_idex = self.children_map_opt[indices]
            tree_len = len(parent_indexs[indices[0]])

            for x_count, x_index in enumerate(parent_indexs[indices[0]]):

                if x_count == tree_len - 1:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......

                    DR_sw = cos * K + sin * torch.matmul(K, K)

                    # rot_mat_spin = rot_mat_twist[:, [x_index]]
                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_1 = torch.matmul(DR_sw, rot_mat_spin)      #
                    # R_derivative[indices, x_count] = torch.matmul(DR_sw, rot_mat_spin)
                    R_derivative[indices, indices] = torch.matmul(DR_sw,
                                                                  rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # DR_ = torch.matmul( rot_mat_chain[:, self.parents[indices]], R_derivative[indices, x_count])
                    DR_ = torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, indices])

                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    # q_derivative[self.children_map_opt[indices], x_count] =

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[indices[0]]] = Dq_1[:, 0, :, 0]
                    # print(Dq_1)

                elif x_count == tree_len - 2:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    #  DR_1_1  === DR_k_1_k_1    ==== DR_(k-1)_(k-1)
                    # parents_1 = self.parents[indices]         ## 父节点
                    #
                    # parents_2 = self.parents[self.parents[indices]]   ## 父节点的父节点

                    # DR_k_1_k_1 = R_derivative[parent_indexs[indices[0]][0], x_count]
                    DR_k_1_k_1 = R_derivative[parents_1, parents_1]

                    # Temp_derivative = torch.matmul(DR_k_1_k_1.transpose(1, 2) , rot_mat_chain[:, self.parents[self.parents[indices]]].transpose(2, 3))

                    # Temp_derivative = torch.matmul(DR_k_1_k_1.transpose(1, 2), rot_mat_chain[:, parents_2].transpose(2, 3))
                    Temp_derivative = torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_1)

                    # orig_vec_derivative = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    orig_vec_inv = torch.matmul(Temp_derivative.transpose(2, 3), orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    # w = torch.cross(child_rest_loc, orig_vec, dim=2)
                    # w_norm = torch.norm(w, dim=2, keepdim=True)

                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K.transpose(2, 3), Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    # Dn_kxx2 = torch.matmul(Dn_k , w.transpose(2, 3)) +  torch.matmul( w, Dn_k.transpose(2, 3)) - ( torch.matmul( Dn_k.transpose(2, 3),w) + torch.matmul( w.transpose(2, 3),Dn_k) ) * ident

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]
                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )

                    # R_derivative[indices, x_count] = torch.matmul(DR_sw_k, rot_mat_spin)
                    R_derivative[indices, parents_1] = torch.matmul(DR_sw_k,
                                                                    rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    q_idex_child = self.parents[q_idex][0]
                    # Dq_k_1_k_1 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1), index_24_to_18[x_index]].unsqueeze(-1).unsqueeze(0)
                    Dq_k_1_k_1 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_1]].unsqueeze(-1).unsqueeze(0)
                    # Temp_q = torch.matmul(torch.matmul(rot_mat_chain[:, self.parents[self.parents[indices]]], DR_k_1_k_1), rot_mat_local[:, indices])  + \
                    #          torch.matmul(rot_mat_chain[:, self.parents[indices]], R_derivative[indices, x_count])
                    # Temp_q = torch.matmul(torch.matmul(rot_mat_chain[:, self.parents[self.parents[indices]]], DR_k_1_k_1), rot_mat_local[:, indices])  + \
                    #          torch.matmul(rot_mat_chain[:, self.parents[indices]], R_derivative[indices, parents_1])
                    Temp_q = torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_1),
                                          rot_mat_local[:, indices]) + \
                             torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, parents_1])

                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_1]] = Dq_k_k_1[:, 0, :, 0]

                    # print(Dq_k_k_1)
                    aaa = 1

                elif x_count == tree_len - 3:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    # orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    DR_k_2_k_2 = R_derivative[parents_2, parents_2]
                    DR_k_1_k_2 = R_derivative[parents_1, parents_2]

                    Temp_derivative = torch.matmul(DR_k_1_k_2.transpose(1, 2),
                                                   rot_mat_chain[:, parents_2].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(rot_mat_local[:, parents_1].transpose(2, 3),
                                                                DR_k_2_k_2.transpose(1, 2)),
                                                   rot_mat_chain[:, parents_3].transpose(2, 3))

                    # orig_vec_derivative = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    R_derivative[indices, parents_2] = torch.matmul(DR_sw_k, rot_mat_spin)

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_2 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_2]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])
                    Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2),
                                                       rot_mat_local[:, parents_1]), rot_mat_local[:, indices]) + \
                             torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2),
                                          rot_mat_local[:, indices]) + \
                             torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, parents_2])
                    ## R_derivative[indices, x_count]  相当于 DR_3_1

                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_2]] = Dq_k_k_2[:, 0, :, 0]
                    # print(Dq_k_k_2)

                elif x_count == tree_len - 4:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[parents_3, parents_3]
                    DR_k_2_k_3 = R_derivative[parents_2, parents_3]
                    DR_k_1_k_3 = R_derivative[parents_1, parents_3]

                    Temp_derivative = torch.matmul(DR_k_1_k_3.transpose(1, 2),
                                                   rot_mat_chain[:, parents_2].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(rot_mat_local[:, parents_1].transpose(2, 3),
                                                                DR_k_2_k_3.transpose(1, 2)),
                                                   rot_mat_chain[:, parents_3].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(
                                          torch.matmul(rot_mat_local[:, parents_1].transpose(2, 3),
                                                       rot_mat_local[:, parents_2].transpose(2, 3)),
                                          DR_k_3_k_3.transpose(1, 2)), rot_mat_chain[:, parents_4].transpose(2, 3))

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[indices, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_3 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_3]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    Temp_q = torch.matmul(torch.matmul(
                        torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3),
                                     rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]),
                                          rot_mat_local[:, indices]) + \
                             torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2),
                                                       rot_mat_local[:, parents_1]), rot_mat_local[:, indices]) + \
                             torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),
                                          rot_mat_local[:, indices]) + \
                             torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, parents_3])

                    ## R_derivative[indices, x_count]  相当于 DR_3_1

                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_3]] = Dq_k_k_3[:, 0, :, 0]
                    # print( Dq_k_k_3)

                    aaa = 1

                elif x_count == tree_len - 5:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[parents_4, parents_4]
                    DR_k_3_k_4 = R_derivative[parents_3, parents_4]
                    DR_k_2_k_4 = R_derivative[parents_2, parents_4]
                    DR_k_1_k_4 = R_derivative[parents_1, parents_4]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_1] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, parents_2] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, parents_3] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, parents_4] = DR_k_4_k_4

                    rot_mat_withDR1 = rot_mat_local_withDR1[:, parents_5:parents_5 + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, parents_5:parents_5 + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, parents_5:parents_5 + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, parents_5:parents_5 + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[indices, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, indices] = R_derivative[indices, parents_4]

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_5:parents_5 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_4 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_4]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_4]] = Dq_k_k_4[:, 0, :, 0]

                    # print(Dq_k_k_4)

                    aaa = 1

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    # orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    # parents_1 = self.parents[indices]         ## 父节点
                    # parents_2 = self.parents[self.parents[indices]]   ## 父节点的父节点
                    # parents_3 =  self.parents[self.parents[self.parents[indices]]]   ## 父节点的父节点的父节点
                    # parents_4 =  self.parents[self.parents[self.parents[self.parents[indices]]]] ## 父节点的父节点的父节点的父节点
                    #
                    # parents_1 = parent_indexs[indices[0]][-2] ## 父节点
                    # parents_2 = parent_indexs[indices[0]][-3] ## 父节点的父节点
                    # parents_3 = parent_indexs[indices[0]][-4] ## 父节点的父节点的父节点
                    # parents_4 = parent_indexs[indices[0]][-5] ## 父节点的父节点的父节点的父节点
                    # parents_5 = parent_indexs[indices[0]][-6]  ## 父节点的父节点的父节点的父节点
                    # parents_6 = self.parents[parents_5]       ## 父节点的父节点的父节点的父节点的父节点

                    DR_k_5_k_5 = R_derivative[parents_5, parents_5]
                    DR_k_4_k_5 = R_derivative[parents_4, parents_5]
                    DR_k_3_k_5 = R_derivative[parents_3, parents_5]
                    DR_k_2_k_5 = R_derivative[parents_2, parents_5]
                    DR_k_1_k_5 = R_derivative[parents_1, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()
                    rot_mat_local_withDR5 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_1] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, parents_2] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, parents_3] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, parents_4] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, parents_5] = DR_k_5_k_5

                    rot_mat_withDR1 = rot_mat_local_withDR1[:, parents_6:parents_6 + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, parents_6:parents_6 + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, parents_6:parents_6 + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, parents_6:parents_6 + 1]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, parents_6:parents_6 + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[indices, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, indices] = R_derivative[indices, parents_5]

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_6:parents_6 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_5 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_5]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_5]] = Dq_k_k_4[:, 0, :, 0]

                    # print(Dq_k_k_4)

                    aaa = 1

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[parents_6, parents_6]
                    DR_k_5_k_6 = R_derivative[parents_5, parents_6]
                    DR_k_4_k_6 = R_derivative[parents_4, parents_6]
                    DR_k_3_k_6 = R_derivative[parents_3, parents_6]
                    DR_k_2_k_6 = R_derivative[parents_2, parents_6]
                    DR_k_1_k_6 = R_derivative[parents_1, parents_6]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()
                    rot_mat_local_withDR5 = rot_mat_local.clone()
                    rot_mat_local_withDR6 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_1] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, parents_2] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, parents_3] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, parents_4] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, parents_5] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, parents_6] = DR_k_6_k_6

                    rot_mat_withDR1 = rot_mat_local_withDR1[:, parents_7:parents_7 + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, parents_7:parents_7 + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, parents_7:parents_7 + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, parents_7:parents_7 + 1]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, parents_7:parents_7 + 1]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, parents_7:parents_7 + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) \
                                      + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                                    3) + rot_mat_withDR6.transpose(
                        2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    # w = axis
                    #
                    # w_norm = axis_norm
                    #
                    # axis = axis / (axis_norm + 1e-8)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[indices, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, indices] = R_derivative[indices, parents_6]

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_7:parents_7 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r: index_r + 1])

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_6 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_6]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_6 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_6]] = Dq_k_k_6[:, 0, :, 0]

                    # print(Dq_k_k_5)
                    aaa = 1


                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_7_k_7 = R_derivative[parents_7, parents_7]
                    DR_k_6_k_7 = R_derivative[parents_6, parents_7]
                    DR_k_5_k_7 = R_derivative[parents_5, parents_7]
                    DR_k_4_k_7 = R_derivative[parents_4, parents_7]
                    DR_k_3_k_7 = R_derivative[parents_3, parents_7]
                    DR_k_2_k_7 = R_derivative[parents_2, parents_7]
                    DR_k_1_k_7 = R_derivative[parents_1, parents_7]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()
                    rot_mat_local_withDR5 = rot_mat_local.clone()
                    rot_mat_local_withDR6 = rot_mat_local.clone()
                    rot_mat_local_withDR7 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_1] = DR_k_1_k_7
                    rot_mat_local_withDR2[:, parents_2] = DR_k_2_k_7
                    rot_mat_local_withDR3[:, parents_3] = DR_k_3_k_7
                    rot_mat_local_withDR4[:, parents_4] = DR_k_4_k_7
                    rot_mat_local_withDR5[:, parents_5] = DR_k_5_k_7
                    rot_mat_local_withDR6[:, parents_6] = DR_k_6_k_7
                    rot_mat_local_withDR7[:, parents_7] = DR_k_7_k_7

                    rot_mat_withDR1 = rot_mat_local_withDR1[:, parents_8:parents_8 + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, parents_8:parents_8 + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, parents_8:parents_8 + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, parents_8:parents_8 + 1]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, parents_8:parents_8 + 1]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, parents_8:parents_8 + 1]
                    rot_mat_withDR7 = rot_mat_local_withDR7[:, parents_8:parents_8 + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR7 = torch.matmul(rot_mat_withDR7, rot_mat_local_withDR7[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) \
                                      + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                                    3) + rot_mat_withDR6.transpose(
                        2, 3) + rot_mat_withDR7.transpose(2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    # w = axis
                    #
                    # w_norm = axis_norm
                    #
                    # axis = axis / (axis_norm + 1e-8)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[indices, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, indices] = R_derivative[indices, parents_7]

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_8:parents_8 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r: index_r + 1])
                    rot_mat_withDR7 = torch.matmul(rot_mat_withDR7, rot_mat_local_withDR7[:, index_r: index_r + 1])

                    q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_7 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_7]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6 + rot_mat_withDR7
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_7 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_7

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_7]] = Dq_k_k_7[:, 0, :, 0]

                    # print(Dq_k_k_5)
                    aaa = 1

        return jacobian

    def forward_jacobian_v2(self,
                            pose_axis_angle,
                            pose_skeleton,
                            rest_J,
                            phis,
                            global_orient,
                            transl=None,
                            leaf_thetas=None,
                            rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        # phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        # rot_mat_chain = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3).repeat(batch_size, 24, 1,
        #                                                                                          1)
        # rot_mat_local = rot_mat_chain.clone()
        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 72, 18], dtype=torch.float32, device=device)
        # R_derivative = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3).repeat(24, 24, 1, 1)
        R_derivative = torch.zeros((24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1

        # q_derivative = torch.zeros((24, 18, 3, 1), dtype=torch.float32,
        #                            device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1

        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)

        index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1,
                          -1]  # 受到影响的父节点的数量

        # parent_indexs = [
        #     [-1],  # 1
        #     [1],  # 1
        #     [2],  # 2
        #     [3],  # 3
        #     [1, 4],  # 4
        #     [2, 5],  # 5
        #     [3, 6],  # 6
        #     [1, 4, 7],  # 7
        #     [2, 5, 8],  # 8
        #     [3, 6, 9],  # 9
        #     [1, 4, 7, 10],  # 10
        #     [2, 5, 8, 11],  # 11
        #     [3, 6, 9, 12],  # 12
        #     [3, 6, 9, 13],  # 13
        #     [3, 6, 9, 14],  # 14
        #     [3, 6, 9, 12, 15],  # 15
        #     [3, 6, 9, 13, 16],  # 16
        #     [3, 6, 9, 14, 17],  # 17
        #     [3, 6, 9, 13, 16, 18],  # 18
        #     [3, 6, 9, 14, 17, 19],  # 19
        #     [3, 6, 9, 13, 16, 18, 20],  # 20
        #     [3, 6, 9, 14, 17, 19, 21],  # 21
        #     [3, 6, 9, 13, 16, 18, 20, 22],  # 22
        #     [3, 6, 9, 14, 17, 19, 21, 23],  # 23
        # ]  # 受到影响的父节点 index

        parent_indexs = [
            [-1],  # 0
            [-1],  # 1
            [-1],  # 2
            [-1],  # 3
            [1],  # 4
            [2],  # 5
            [3],  # 6
            [1, 4],  # 7
            [2, 5],  # 8
            [3, 6],  # 9
            [1, 4, 7],  # 10
            [2, 5, 8],  # 11
            [3, 6, 9],  # 12
            [3, 6, 9],  # 13
            [3, 6, 9],  # 14
            [3, 6, 9, 12],  # 15
            [3, 6, 9, 13],  # 16
            [3, 6, 9, 14],  # 17
            [3, 6, 9, 13, 16],  # 18
            [3, 6, 9, 14, 17],  # 19
            [3, 6, 9, 13, 16, 18],  # 20
            [3, 6, 9, 14, 17, 19],  # 21
            [3, 6, 9, 13, 16, 18, 20],  # 22
            [3, 6, 9, 14, 17, 19, 21]  # 23
        ]  # 受到影响的父节点 index

        # idx_jacobian = [
        #     [3],  # 1
        #     [1], [2], [12], [13], [14],  # 4
        #     [4], [5], [16], [17],  # 5
        #     [7], [8], [18], [19],  # 6
        #     [20], [21],  # 7
        #     ]  #少了 0,10,11,15,22,23
        # idx_jacobian = [
        #     [1], [2],  # 4
        #     [4], [5],  # 5
        #     [7], [8],  # 6
        #     ]  #少了 0,10,11,15,22,23
        #  [16], [17], [18], [19], [20], [21]
        # idx_jacobian = [
        #     [1], [2],  # 4
        #     [4], [5],  # 5
        #     [7], [8],  # 6
        #     [3], [6], [9], [12],
        #     [13], [14], [16], [17], [18], [19], [20], [21]]  # 少了 0,10,11,15,22,23
        # idx_jacobian = [
        #     [4], [5],
        #     [7], [8],  # 6
        #     [10], [11],
        #     [6], [9], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]
        #     ]  # 少了 0,10,11,15,22,23
        # idx_jacobian = [
        #     [4], [5],
        #     [7], [8],  # 6
        #     [10], [11],
        #     [6], [9], [12], [13], [14], [15], [16],[17], [18], [19], [20], [21], [22], [23]
        #     ]  # 少了 0,10,11,15,22,23
        idx_jacobian = [
            [4], [5],
            [7], [8],  # 6
            [10], [11],
            [6], [9], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]
        ]  # 少了 0,10,11,15,22,23

        ### bug 出现在 [22], [23] ！！！！！！！！！！！！！！

        ## 第九个节点有三个字节点，所以index 9 的时候如果计算三次

        ## 我应该按着树状图的节点的输出循环开始往下写，每个节点输出都有对应的影响到的输入x，这样代码效率最高

        for idx_lev in range(len(idx_jacobian)):

            indices = idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            parents_8 = self.parents[parents_7]  ## 父节点的父节点的父节点的父节点
            # orig_vec_unrotate = rel_pose_skeleton[:, self.children_map_opt[indices]]
            orig_vec_unrotate = rel_pose_skeleton[:, indices]

            # if idx_lev == 8:
            #     orig_vec_unrotate = 0 * orig_vec_unrotate

            if indices == [12] or indices == [13] or indices == [14]:
                orig_vec_unrotate = 0 * orig_vec_unrotate

            orig_vec = torch.matmul(
                rot_mat_chain[:, parents_2].transpose(2, 3),
                orig_vec_unrotate
            )
            if indices == [10]:
                aaa = 1

            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?

            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=2)
            w_norm = torch.norm(w, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain[:, parents_1] = torch.matmul(
                rot_mat_chain[:, parents_2],
                rot_mat)
            rot_mat_local[:, parents_1] = rot_mat

            # q_idex = self.children_map_opt[indices]
            q_idex = indices[0]
            q_idex_child = self.parents[q_idex]
            tree_len = len(parent_indexs[indices[0]])

            for x_count, x_index in enumerate(parent_indexs[indices[0]]):

                if x_count == tree_len - 1:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[parents_1, parents_1] = torch.matmul(DR_sw,
                                                                      rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[:, parents_2], R_derivative[parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_1]] = Dq_1[:, 0, :, 0]

                elif x_count == tree_len - 2:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[parents_2, parents_2]

                    # Temp_derivative = torch.matmul(DR_k_1_k_1.transpose(1, 2) , rot_mat_chain[:, self.parents[self.parents[indices]]].transpose(2, 3))

                    # Temp_derivative = torch.matmul(DR_k_1_k_1.transpose(1, 2), rot_mat_chain[:, parents_2].transpose(2, 3))
                    Temp_derivative = torch.matmul(rot_mat_chain[:, parents_3], DR_k_1_k_1)

                    # orig_vec_derivative = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    orig_vec_inv = torch.matmul(Temp_derivative.transpose(2, 3), orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    # w = torch.cross(child_rest_loc, orig_vec, dim=2)
                    # w_norm = torch.norm(w, dim=2, keepdim=True)

                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2).view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # R_derivative[indices, x_count] = torch.matmul(DR_sw_k, rot_mat_spin)
                    R_derivative[parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                      rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # Dq_k_1_k_1 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1), index_24_to_18[x_index]].unsqueeze(-1).unsqueeze(0)
                    Dq_k_1_k_1 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_2]].unsqueeze(-1).unsqueeze(0)

                    Temp_q = torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_1_k_1),
                                          rot_mat_local[:, parents_1]) + \
                             torch.matmul(rot_mat_chain[:, parents_2], R_derivative[parents_1, parents_2])

                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_2]] = Dq_k_k_1[:, 0, :, 0]

                elif x_count == tree_len - 3:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    # orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    DR_k_2_k_2 = R_derivative[parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[parents_2, parents_3]

                    Temp_derivative = torch.matmul(DR_k_1_k_2.transpose(1, 2),
                                                   rot_mat_chain[:, parents_3].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(rot_mat_local[:, parents_2].transpose(2, 3),
                                                                DR_k_2_k_2.transpose(1, 2)),
                                                   rot_mat_chain[:, parents_4].transpose(2, 3))

                    # orig_vec_derivative = rel_pose_skeleton[:, self.children_map_opt[indices]]

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    R_derivative[parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    # q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_2 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_3]].unsqueeze(-1).unsqueeze(0)

                    Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_2_k_2),
                                                       rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                             torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_1_k_2),
                                          rot_mat_local[:, parents_1]) + \
                             torch.matmul(rot_mat_chain[:, parents_2], R_derivative[parents_1, parents_3])
                    ## R_derivative[indices, x_count]  相当于 DR_3_1

                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_3]] = Dq_k_k_2[:, 0, :, 0]
                    # print(Dq_k_k_2)

                elif x_count == tree_len - 4:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[parents_2, parents_4]

                    Temp_derivative = torch.matmul(DR_k_1_k_3.transpose(1, 2),
                                                   rot_mat_chain[:, parents_3].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(rot_mat_local[:, parents_2].transpose(2, 3),
                                                                DR_k_2_k_3.transpose(1, 2)),
                                                   rot_mat_chain[:, parents_4].transpose(2, 3)) + \
                                      torch.matmul(torch.matmul(
                                          torch.matmul(rot_mat_local[:, parents_2].transpose(2, 3),
                                                       rot_mat_local[:, parents_3].transpose(2, 3)),
                                          DR_k_3_k_3.transpose(1, 2)), rot_mat_chain[:, parents_5].transpose(2, 3))

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)

                    # q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_3 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_4]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    Temp_q = torch.matmul(torch.matmul(
                        torch.matmul(torch.matmul(rot_mat_chain[:, parents_5], DR_k_3_k_3),
                                     rot_mat_local[:, parents_3]), rot_mat_local[:, parents_2]),
                        rot_mat_local[:, parents_1]) + \
                             torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_2_k_3),
                                                       rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                             torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_1_k_3),
                                          rot_mat_local[:, parents_1]) + \
                             torch.matmul(rot_mat_chain[:, parents_2], R_derivative[parents_1, parents_4])

                    ## R_derivative[indices, x_count]  相当于 DR_3_1

                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_4]] = Dq_k_k_3[:, 0, :, 0]
                    # print( Dq_k_k_3)

                    aaa = 1

                elif x_count == tree_len - 5:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[parents_2, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_2] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, parents_3] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, parents_4] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, parents_5] = DR_k_4_k_4

                    # rot_mat_withDR1 = rot_mat_local_withDR1[:, parents_6:parents_6 + 1]
                    # rot_mat_withDR2 = rot_mat_local_withDR2[:, parents_6:parents_6 + 1]
                    # rot_mat_withDR3 = rot_mat_local_withDR3[:, parents_6:parents_6 + 1]
                    # rot_mat_withDR4 = rot_mat_local_withDR4[:, parents_6:parents_6 + 1]

                    ancestor = self.parents[parent_indexs[indices[0]][0]]
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, ancestor:ancestor + 1]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, ancestor:ancestor + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, ancestor:ancestor + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, ancestor:ancestor + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, ancestor:ancestor + 1]

                    # rot_mat_withDR0 = rot_mat_chain[:, parents_6:parents_6 + 1].clone()
                    # rot_mat_withDR1 = rot_mat_chain[:, parents_6:parents_6 + 1].clone()
                    # rot_mat_withDR2 = rot_mat_chain[:, parents_6:parents_6 + 1].clone()
                    # rot_mat_withDR3 = rot_mat_chain[:, parents_6:parents_6 + 1].clone()
                    # rot_mat_withDR4 = rot_mat_chain[:, parents_6:parents_6 + 1].clone()

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, parents_1] = R_derivative[parents_1, parents_5]

                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_6:parents_6 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])

                    # q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_4 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_5]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_5]] = Dq_k_k_4[:, 0, :, 0]

                    # print(Dq_k_k_4)

                    aaa = 1

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[parents_2, parents_6]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()
                    rot_mat_local_withDR5 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_2] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, parents_3] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, parents_4] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, parents_5] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, parents_6] = DR_k_5_k_5

                    ancestor = self.parents[parent_indexs[indices[0]][0]]
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, ancestor:ancestor + 1]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, ancestor:ancestor + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, ancestor:ancestor + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, ancestor:ancestor + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, ancestor:ancestor + 1]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, ancestor:ancestor + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, parents_1]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, parents_1] = R_derivative[parents_1, parents_6]

                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_7:parents_7 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])

                    # q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_5 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_6]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_6]] = Dq_k_k_4[:, 0, :, 0]

                    # print(Dq_k_k_4)

                    aaa = 1

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[parents_2, parents_7]

                    rot_mat_local_withDR0 = rot_mat_local.clone()
                    rot_mat_local_withDR1 = rot_mat_local.clone()
                    rot_mat_local_withDR2 = rot_mat_local.clone()
                    rot_mat_local_withDR3 = rot_mat_local.clone()
                    rot_mat_local_withDR4 = rot_mat_local.clone()
                    rot_mat_local_withDR5 = rot_mat_local.clone()
                    rot_mat_local_withDR6 = rot_mat_local.clone()

                    rot_mat_local_withDR1[:, parents_2] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, parents_3] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, parents_4] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, parents_5] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, parents_6] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, parents_7] = DR_k_6_k_6

                    ancestor = self.parents[parent_indexs[indices[0]][0]]
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, ancestor:ancestor + 1]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, ancestor:ancestor + 1]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, ancestor:ancestor + 1]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, ancestor:ancestor + 1]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, ancestor:ancestor + 1]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, ancestor:ancestor + 1]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, ancestor:ancestor + 1]

                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, index_r: index_r + 1])
                    for index_r in parent_indexs[indices[0]][:-1]:
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                       rot_mat_local_withDR6[:, index_r: index_r + 1])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                            3) + rot_mat_withDR6.transpose(
                        2, 3)

                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)

                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)

                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    # Dn_kxx2 = 2 * torch.matmul( K, Dn_kx)
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)

                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2

                    # rot_mat_spin = rot_mat_twist[:, indices]

                    # DR_2 = torch.matmul( DR_sw_2 , rot_mat_spin )
                    R_derivative[parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    rot_mat_local_withDR0[:, parents_1] = R_derivative[parents_1, parents_7]

                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, parents_8:parents_8 + 1]

                    for index_r in parent_indexs[indices[0]]:
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, index_r: index_r + 1])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r: index_r + 1])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r: index_r + 1])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r: index_r + 1])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r: index_r + 1])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r: index_r + 1])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r: index_r + 1])

                    # q_idex_child = self.parents[q_idex][0]

                    Dq_k_1_k_6 = jacobian[:, 3 * q_idex_child: 3 * (q_idex_child + 1),
                                 index_24_to_18[parents_7]].unsqueeze(-1).unsqueeze(0)

                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_2), rot_mat_local[:, parents_2]   ), rot_mat_local[:, parents_1]  ) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_2), rot_mat_local[:, parents_1]   ) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1], R_derivative[indices, x_count])

                    #  这部分改成跟上面的一样 ， 需要差乘最后一项而已

                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    # Temp_q = torch.matmul(torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_4], DR_k_3_k_3), rot_mat_local[:, parents_3]   ), rot_mat_local[:, parents_2]  ), rot_mat_local[:, parents_1] ) +\
                    #          torch.matmul(torch.matmul(torch.matmul(rot_mat_chain[:, parents_3], DR_k_2_k_3),  rot_mat_local[:, parents_2]), rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(torch.matmul(rot_mat_chain[:, parents_2], DR_k_1_k_3),  rot_mat_local[:, parents_1]) + \
                    #          torch.matmul(rot_mat_chain[:, parents_1],  R_derivative[indices, x_count])

                    ## R_derivative[indices, x_count]  相当于 DR_5_1

                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6

                    jacobian[:, 3 * q_idex: 3 * (q_idex + 1), index_24_to_18[parents_7]] = Dq_k_k_5[:, 0, :, 0]

                    # print(Dq_k_k_5)
                    aaa = 1

        return jacobian

    def forward_jacobian_v2_batch(self,
                                  pose_axis_angle,
                                  pose_skeleton,
                                  rest_J,
                                  global_orient,
                                  rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        for idx_lev in range(len(self.idx_jacobian)):
            indices = self.idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 3:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[:, parents_2].transpose(2, 3),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=2)
            w_norm = torch.norm(w, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            rot_mat_chain[:, parents_1] = torch.matmul(
                rot_mat_chain[:, parents_2],
                rot_mat)
            rot_mat_local[:, parents_1] = rot_mat

            q_idex = indices
            q_idex_child = self.parents[q_idex]
            tree_len = len(self.parent_indexs[indices[0]])
            # parent_index =  [self.parent_indexs[indices[i]] for i in range(tree_len)]
            parent_index = torch.tensor([self.parent_indexs[indices[i]] for i in range(len_indices)])

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[:, parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_1
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2).view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_2]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_2
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_2

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_3]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_3
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_3
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_3

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_4]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_4

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_5]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_5
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_6]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, :, -7] = DR_k_6_k_6
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                       rot_mat_local_withDR6[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                            3) + rot_mat_withDR6.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_7]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, :, index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, :, 0]

        return jacobian.reshape(batch_size, 18, 24 * 3).transpose(1, 2)

    def forward_jacobian_and_pred(self,
                                  pose_axis_angle,
                                  pose_skeleton,
                                  rest_J,
                                  phis,
                                  global_orient,
                                  transl=None,
                                  leaf_thetas=None,
                                  rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        for idx_lev in range(len(self.idx_jacobian)):
            indices = self.idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[:, parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 3:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[:, parents_2].transpose(2, 3),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=2)
            w_norm = torch.norm(w, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            rot_mat_chain[:, parents_1] = torch.matmul(
                rot_mat_chain[:, parents_2],
                rot_mat)
            rot_mat_local[:, parents_1] = rot_mat

            q_idex = indices
            q_idex_child = self.parents[q_idex]
            tree_len = len(self.parent_indexs[indices[0]])
            # parent_index =  [self.parent_indexs[indices[i]] for i in range(tree_len)]
            parent_index = torch.tensor([self.parent_indexs[indices[i]] for i in range(len_indices)])

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[:, parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_1
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2).view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_2]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_2
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_2

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_3]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_3
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_3
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_3

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_4]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_4

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_5]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_5
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_6]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, :, -7] = DR_k_6_k_6
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                       rot_mat_local_withDR6[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                            3) + rot_mat_withDR6.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_7]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, :, index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, :, 0]

        leaf_index = [15, 22, 23, 10, 11]
        rotate_rest_pose[:, leaf_index] = rotate_rest_pose[:, self.parents[leaf_index]] + torch.matmul(
            rot_mat_chain[:, self.parents[leaf_index]],
            rel_rest_pose[:, leaf_index]
        )
        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return jacobian.reshape(batch_size, 18, 24 * 3).transpose(1, 2), new_joints

    def forward_jacobian_and_pred_v2(self,
                                     pose_axis_angle,
                                     pose_skeleton,
                                     rest_J,
                                     global_orient,
                                     rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        for idx_lev in range(len(self.idx_jacobian)):
            indices = self.idx_jacobian[idx_lev]
            len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[:, parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 3:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[:, parents_2].transpose(2, 3),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=2)
            w_norm = torch.norm(w, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            rot_mat_chain[:, parents_1] = torch.matmul(
                rot_mat_chain[:, parents_2],
                rot_mat)
            rot_mat_local[:, parents_1] = rot_mat

            q_idex = indices
            q_idex_child = self.parents[q_idex]
            tree_len = len(self.parent_indexs[indices[0]])
            # parent_index =  [self.parent_indexs[indices[i]] for i in range(tree_len)]
            parent_index = torch.tensor([self.parent_indexs[indices[i]] for i in range(len_indices)])

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[:, parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_1
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2).view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_2]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_2
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_2

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_3]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_3
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_3
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_3

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_4]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]

                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_4
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_4
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_4
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_4

                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_5]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_5
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_5
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_5
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_5
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_5
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_6]
                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                       rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                    rot_mat_local_withDR1[:, :, -2] = DR_k_1_k_6
                    rot_mat_local_withDR2[:, :, -3] = DR_k_2_k_6
                    rot_mat_local_withDR3[:, :, -4] = DR_k_3_k_6
                    rot_mat_local_withDR4[:, :, -5] = DR_k_4_k_6
                    rot_mat_local_withDR5[:, :, -6] = DR_k_5_k_6
                    rot_mat_local_withDR6[:, :, -7] = DR_k_6_k_6
                    rot_mat_withDR0 = rot_mat_local_withDR0[:, :, 0]
                    rot_mat_withDR1 = rot_mat_local_withDR1[:, :, 0]
                    rot_mat_withDR2 = rot_mat_local_withDR2[:, :, 0]
                    rot_mat_withDR3 = rot_mat_local_withDR3[:, :, 0]
                    rot_mat_withDR4 = rot_mat_local_withDR4[:, :, 0]
                    rot_mat_withDR5 = rot_mat_local_withDR5[:, :, 0]
                    rot_mat_withDR6 = rot_mat_local_withDR6[:, :, 0]

                    for index_r in range(1, tree_len - 1):
                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                       rot_mat_local_withDR1[:, :, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                       rot_mat_local_withDR2[:, :, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                       rot_mat_local_withDR3[:, :, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                       rot_mat_local_withDR4[:, :, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                       rot_mat_local_withDR5[:, :, index_r])
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                       rot_mat_local_withDR6[:, :, index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(2, 3) + rot_mat_withDR2.transpose(2,
                                                                                                  3) + rot_mat_withDR3.transpose(
                        2, 3) + rot_mat_withDR4.transpose(2, 3) + rot_mat_withDR5.transpose(2,
                                                                                            3) + rot_mat_withDR6.transpose(
                        2, 3)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=2)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(2, 3)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=2)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=2) \
                        .view((batch_size, len_indices, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)
                    rot_mat_local_withDR0[:, :, -1] = R_derivative[:, parents_1, parents_7]

                    for index_r in range(1, tree_len):
                        rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, :, index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, :, index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, :, index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, :, index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, :, index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, :, index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, :, index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, :, 0]

        leaf_index = [15, 22, 23, 10, 11]
        rotate_rest_pose[:, leaf_index] = rotate_rest_pose[:, self.parents[leaf_index]] + torch.matmul(
            rot_mat_chain[:, self.parents[leaf_index]],
            rel_rest_pose[:, leaf_index]
        )
        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return jacobian.reshape(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mat_local

    def forward_jacobian_and_pred_v2_train(self,  ### 这份程序有BUG，但是先放着，写V3
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])

        # idx_jacobian = [
        #     [4, 5, 6],
        #     [7, 8, 9],
        #     [10, 11],
        #     [12, 13, 14],
        #     [15, 16, 17],
        #     [18, 19],
        #     [20, 21],
        #     [22, 23]
        # ]  # 少了0,1,2,3

        parent_indexs = [
            [-1],  # 0
            [0, 1],  # 1
            [0, 2],  # 2
            [0, 3],  # 3
            [0, 1, 4],  # 4
            [0, 2, 5],  # 5
            [0, 3, 6],  # 6
            [0, 1, 4, 7],  # 7
            [0, 2, 5, 7],  # 8
            [0, 3, 6, 9],  # 9
            [0, 1, 4, 7, 10],  # 10
            [0, 2, 5, 8, 11],  # 11
            [0, 3, 6, 9, 12],  # 12
            [0, 3, 6, 9, 13],  # 13
            [0, 3, 6, 9, 14],  # 14
            [0, 3, 6, 9, 12, 15],  # 15
            [0, 3, 6, 9, 13, 16],  # 16
            [0, 3, 6, 9, 14, 17],  # 17
            [0, 3, 6, 9, 13, 16, 18],  # 18
            [0, 3, 6, 9, 14, 17, 19],  # 19
            [0, 3, 6, 9, 13, 16, 18, 20],  # 20
            [0, 3, 6, 9, 14, 17, 19, 21],  # 21
            [0, 3, 6, 9, 13, 16, 18, 20, 22],  # 22
            [0, 3, 6, 9, 14, 17, 19, 21, 23]  # 23
        ]  # 受到影响的父节点 index

        leaf_cnt = 0
        for idx_lev in range(1, 24):
            indices = idx_lev
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            # parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                rot_mat_chain[:, parents_1],
                rel_rest_pose[:, indices]
            )

            if children_1 == -1:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, parents_1],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

            elif children_1 == 12:

                # if idx_lev == 9 :
                #     orig_vec_unrotate = 0 * orig_vec_unrotate
                aaa = 1
                orig_vec_unrotate = rel_pose_skeleton[:, children_1]
                orig_vec_unrotate = 0 * orig_vec_unrotate

                # if idx_lev == 9 :
                #     orig_vec_unrotate = 0 * orig_vec_unrotate

                orig_vec = torch.matmul(
                    rot_mat_chain[:, parents_1].transpose(1, 2),
                    orig_vec_unrotate
                )
                child_rest_loc = rel_rest_pose[:, children_1]  # need rotation back ?

                # (B, K, 3, 1)
                w = torch.cross(child_rest_loc, orig_vec, dim=1)
                w_norm = torch.norm(w, dim=1, keepdim=True)
                # (B, K, 1, 1)
                cos = full_angle[:, indices].cos()
                sin = full_angle[:, indices].sin()
                # (B, K, 3, 1)
                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat_spin = rot_mat_twist[:, indices]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, parents_1],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

                children_three = [12, 13, 14]
                q_idex_child = indices
                tree_len = len(parent_indexs[indices])
                for c in children_three:
                    q_idex = c
                    for x_count in range(tree_len - 1):

                        if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                            DR_sw = cos * K + sin * torch.matmul(K, K)
                            R_derivative[:, indices, indices] = torch.matmul(DR_sw,
                                                                             rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                            DR_ = torch.matmul(rot_mat_chain[:, parents_1], R_derivative[:, indices, indices])
                            Dq_1 = torch.matmul(DR_, child_rest_loc)
                            jacobian[:, index_24_to_18[indices], q_idex] = Dq_1[:, :, 0]

                        elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                            DR_k_1_k_1 = R_derivative[:, parents_1, parents_1]
                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn_k = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3),
                                Dw)

                            rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1).view((batch_size, 3, 3))
                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, indices, parents_1] = torch.matmul(DR_sw_k,
                                                                               rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_1]

                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                               rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_1], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                            Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                            jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_k_k_1[:, :, 0]

                        elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......
                            # 写到这

                            DR_k_2_k_2 = R_derivative[:, parents_2, parents_2]
                            DR_k_1_k_2 = R_derivative[:, parents_1, parents_2]
                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_2
                            rot_mat_local_withDR2[:, -3] = DR_k_2_k_2

                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]

                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])
                                rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                               rot_mat_local_withDR2[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3,
                                Dw)
                            rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1) \
                                .view((batch_size, 3, 3))

                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_3]

                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                               rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])

                            Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                            Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                            jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                        elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                            DR_k_3_k_3 = R_derivative[:, parents_3, parents_3]
                            DR_k_2_k_3 = R_derivative[:, parents_2, parents_3]
                            DR_k_1_k_3 = R_derivative[:, parents_1, parents_3]

                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_3
                            rot_mat_local_withDR2[:, -3] = DR_k_2_k_3
                            rot_mat_local_withDR3[:, -4] = DR_k_3_k_3

                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                            rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]

                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])
                                rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                               rot_mat_local_withDR2[:, index_r])
                                rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                               rot_mat_local_withDR3[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                          2) + rot_mat_withDR3.transpose(
                                1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3,
                                Dw)

                            rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1) \
                                .view((batch_size, 3, 3))

                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, indices, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_3]

                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                               rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                            Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                            Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                            jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_3[:, :, 0]

                        elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                            DR_k_4_k_4 = R_derivative[:, parents_4, parents_4]
                            DR_k_3_k_4 = R_derivative[:, parents_3, parents_4]
                            DR_k_2_k_4 = R_derivative[:, parents_2, parents_4]
                            DR_k_1_k_4 = R_derivative[:, parents_1, parents_4]

                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_4
                            rot_mat_local_withDR2[:, -3] = DR_k_2_k_4
                            rot_mat_local_withDR3[:, -4] = DR_k_3_k_4
                            rot_mat_local_withDR4[:, -5] = DR_k_4_k_4

                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                            rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                            rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])
                                rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                               rot_mat_local_withDR2[:, index_r])
                                rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                               rot_mat_local_withDR3[:, index_r])
                                rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                               rot_mat_local_withDR4[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                          2) + rot_mat_withDR3.transpose(
                                1, 2) + rot_mat_withDR4.transpose(1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3,
                                Dw)
                            rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1) \
                                .view((batch_size, 3, 3))
                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, indices, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_4]
                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                               rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                            Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                            Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                            jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_4[:, :, 0]

                        elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                            DR_k_5_k_5 = R_derivative[:, parents_5, parents_5]
                            DR_k_4_k_5 = R_derivative[:, parents_4, parents_5]
                            DR_k_3_k_5 = R_derivative[:, parents_3, parents_5]
                            DR_k_2_k_5 = R_derivative[:, parents_2, parents_5]
                            DR_k_1_k_5 = R_derivative[:, parents_1, parents_5]
                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_5
                            rot_mat_local_withDR2[:, -3] = DR_k_2_k_5
                            rot_mat_local_withDR3[:, -4] = DR_k_3_k_5
                            rot_mat_local_withDR4[:, -5] = DR_k_4_k_5
                            rot_mat_local_withDR5[:, -6] = DR_k_5_k_5
                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                            rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                            rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                            rot_mat_withDR5 = rot_mat_local_withDR5[:, 0]

                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])
                                rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                               rot_mat_local_withDR2[:, index_r])
                                rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                               rot_mat_local_withDR3[:, index_r])
                                rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                               rot_mat_local_withDR4[:, index_r])
                                rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                               rot_mat_local_withDR5[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                          2) + rot_mat_withDR3.transpose(
                                1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3,
                                Dw)
                            rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1) \
                                .view((batch_size, 3, 3))
                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, indices, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_5]
                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                               rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r])
                            Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                            Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                            jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                        elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                            DR_k_6_k_6 = R_derivative[:, parents_6, parents_6]
                            DR_k_5_k_6 = R_derivative[:, parents_5, parents_6]
                            DR_k_4_k_6 = R_derivative[:, parents_4, parents_6]
                            DR_k_3_k_6 = R_derivative[:, parents_3, parents_6]
                            DR_k_2_k_6 = R_derivative[:, parents_2, parents_6]
                            DR_k_1_k_6 = R_derivative[:, parents_1, parents_6]
                            rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                            rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                            rot_mat_local_withDR1[:, -2] = DR_k_1_k_6
                            rot_mat_local_withDR2[:, -3] = DR_k_2_k_6
                            rot_mat_local_withDR3[:, -4] = DR_k_3_k_6
                            rot_mat_local_withDR4[:, -5] = DR_k_4_k_6
                            rot_mat_local_withDR5[:, -6] = DR_k_5_k_6
                            rot_mat_local_withDR6[:, -7] = DR_k_6_k_6
                            rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                            rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                            rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                            rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                            rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                            rot_mat_withDR5 = rot_mat_local_withDR5[:, 0]
                            rot_mat_withDR6 = rot_mat_local_withDR6[:, 0]

                            for index_r in range(1, tree_len - 1):
                                rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                               rot_mat_local_withDR1[:, index_r])
                                rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                               rot_mat_local_withDR2[:, index_r])
                                rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                               rot_mat_local_withDR3[:, index_r])
                                rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                               rot_mat_local_withDR4[:, index_r])
                                rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                               rot_mat_local_withDR5[:, index_r])
                                rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                               rot_mat_local_withDR6[:, index_r])

                            Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                          2) + rot_mat_withDR3.transpose(
                                1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1,
                                                                                                    2) + rot_mat_withDR6.transpose(
                                1, 2)
                            orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                            Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                            Dn = torch.matmul(
                                (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3,
                                Dw)
                            rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                            Dn_kx = torch.cat(
                                [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial,
                                 rx_partial,
                                 zeros], dim=1) \
                                .view((batch_size, 3, 3))

                            Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                            DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                            R_derivative[:, indices, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                            rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_6]

                            for index_r in range(1, tree_len):
                                rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r])

                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r])
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r])
                            Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                            Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                            Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                            jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_5[:, :, 0]




            else:

                orig_vec_unrotate = rel_pose_skeleton[:, children_1]

                # if idx_lev == 9 :
                #     orig_vec_unrotate = 0 * orig_vec_unrotate

                orig_vec = torch.matmul(
                    rot_mat_chain[:, parents_1].transpose(1, 2),
                    orig_vec_unrotate
                )
                child_rest_loc = rel_rest_pose[:, children_1]  # need rotation back ?

                # (B, K, 3, 1)
                w = torch.cross(child_rest_loc, orig_vec, dim=1)
                w_norm = torch.norm(w, dim=1, keepdim=True)
                # (B, K, 1, 1)
                cos = full_angle[:, indices].cos()
                sin = full_angle[:, indices].sin()
                # (B, K, 3, 1)
                axis = w / (w_norm + 1e-8)
                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=1)
                zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                    .view((batch_size, 3, 3))
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat_spin = rot_mat_twist[:, indices]
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, parents_1],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

                q_idex = children_1
                q_idex_child = indices

                # tree_len = len(self.parent_indexs[children_1])
                tree_len = len(parent_indexs[indices])
                # parent_index =  [self.parent_indexs[indices[i]] for i in range(tree_len)]
                # parent_index = torch.tensor([self.parent_indexs[indices[i]] for i in range(len_indices)])
                # parent_index = torch.tensor(self.parent_indexs[children_1])
                parent_index = torch.tensor(parent_indexs[indices])

                for x_count in range(tree_len - 1):

                    if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                        DR_sw = cos * K + sin * torch.matmul(K, K)
                        R_derivative[:, indices, indices] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                        DR_ = torch.matmul(rot_mat_chain[:, parents_1], R_derivative[:, indices, indices])
                        Dq_1 = torch.matmul(DR_, child_rest_loc)
                        jacobian[:, index_24_to_18[indices], q_idex] = Dq_1[:, :, 0]

                    elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                        DR_k_1_k_1 = R_derivative[:, parents_1, parents_1]
                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn_k = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                        rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1).view((batch_size, 3, 3))
                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, indices, parents_1] = torch.matmul(DR_sw_k,
                                                                           rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_1]

                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                           rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_1], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                        Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                        jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_k_k_1[:, :, 0]

                    elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......
                        # 写到这

                        DR_k_2_k_2 = R_derivative[:, parents_2, parents_2]
                        DR_k_1_k_2 = R_derivative[:, parents_1, parents_2]
                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()

                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_2
                        rot_mat_local_withDR2[:, -3] = DR_k_2_k_2

                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]

                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                           rot_mat_local_withDR2[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                        rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1) \
                            .view((batch_size, 3, 3))

                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_3]

                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                           rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])

                        Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                        Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                        jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                    elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                        DR_k_3_k_3 = R_derivative[:, parents_3, parents_3]
                        DR_k_2_k_3 = R_derivative[:, parents_2, parents_3]
                        DR_k_1_k_3 = R_derivative[:, parents_1, parents_3]

                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()

                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_3
                        rot_mat_local_withDR2[:, -3] = DR_k_2_k_3
                        rot_mat_local_withDR3[:, -4] = DR_k_3_k_3

                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                        rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]

                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                           rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                           rot_mat_local_withDR3[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                      2) + rot_mat_withDR3.transpose(
                            1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                        rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1) \
                            .view((batch_size, 3, 3))

                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, indices, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_3]

                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                           rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                        Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                        Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                        jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_3[:, :, 0]

                    elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                        DR_k_4_k_4 = R_derivative[:, parents_4, parents_4]
                        DR_k_3_k_4 = R_derivative[:, parents_3, parents_4]
                        DR_k_2_k_4 = R_derivative[:, parents_2, parents_4]
                        DR_k_1_k_4 = R_derivative[:, parents_1, parents_4]

                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()

                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_4
                        rot_mat_local_withDR2[:, -3] = DR_k_2_k_4
                        rot_mat_local_withDR3[:, -4] = DR_k_3_k_4
                        rot_mat_local_withDR4[:, -5] = DR_k_4_k_4

                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                        rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                        rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                           rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                           rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                           rot_mat_local_withDR4[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                      2) + rot_mat_withDR3.transpose(
                            1, 2) + rot_mat_withDR4.transpose(1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                        rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1) \
                            .view((batch_size, 3, 3))
                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, indices, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_4]
                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                           rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                        Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                        Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                        jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_4[:, :, 0]

                    elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                        DR_k_5_k_5 = R_derivative[:, parents_5, parents_5]
                        DR_k_4_k_5 = R_derivative[:, parents_4, parents_5]
                        DR_k_3_k_5 = R_derivative[:, parents_3, parents_5]
                        DR_k_2_k_5 = R_derivative[:, parents_2, parents_5]
                        DR_k_1_k_5 = R_derivative[:, parents_1, parents_5]
                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_5
                        rot_mat_local_withDR2[:, -3] = DR_k_2_k_5
                        rot_mat_local_withDR3[:, -4] = DR_k_3_k_5
                        rot_mat_local_withDR4[:, -5] = DR_k_4_k_5
                        rot_mat_local_withDR5[:, -6] = DR_k_5_k_5
                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                        rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                        rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                        rot_mat_withDR5 = rot_mat_local_withDR5[:, 0]

                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                           rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                           rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                           rot_mat_local_withDR4[:, index_r])
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                           rot_mat_local_withDR5[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                      2) + rot_mat_withDR3.transpose(
                            1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                        rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1) \
                            .view((batch_size, 3, 3))
                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, indices, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_5]
                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                                                           rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r])
                        Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                        Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                        jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                    elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                        DR_k_6_k_6 = R_derivative[:, parents_6, parents_6]
                        DR_k_5_k_6 = R_derivative[:, parents_5, parents_6]
                        DR_k_4_k_6 = R_derivative[:, parents_4, parents_6]
                        DR_k_3_k_6 = R_derivative[:, parents_3, parents_6]
                        DR_k_2_k_6 = R_derivative[:, parents_2, parents_6]
                        DR_k_1_k_6 = R_derivative[:, parents_1, parents_6]
                        rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                        rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR2 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR3 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR4 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR5 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR6 = rot_mat_local_withDR0.clone()
                        rot_mat_local_withDR1[:, -2] = DR_k_1_k_6
                        rot_mat_local_withDR2[:, -3] = DR_k_2_k_6
                        rot_mat_local_withDR3[:, -4] = DR_k_3_k_6
                        rot_mat_local_withDR4[:, -5] = DR_k_4_k_6
                        rot_mat_local_withDR5[:, -6] = DR_k_5_k_6
                        rot_mat_local_withDR6[:, -7] = DR_k_6_k_6
                        rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                        rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                        rot_mat_withDR2 = rot_mat_local_withDR2[:, 0]
                        rot_mat_withDR3 = rot_mat_local_withDR3[:, 0]
                        rot_mat_withDR4 = rot_mat_local_withDR4[:, 0]
                        rot_mat_withDR5 = rot_mat_local_withDR5[:, 0]
                        rot_mat_withDR6 = rot_mat_local_withDR6[:, 0]

                        for index_r in range(1, tree_len - 1):
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                                                           rot_mat_local_withDR1[:, index_r])
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2,
                                                           rot_mat_local_withDR2[:, index_r])
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3,
                                                           rot_mat_local_withDR3[:, index_r])
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4,
                                                           rot_mat_local_withDR4[:, index_r])
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5,
                                                           rot_mat_local_withDR5[:, index_r])
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6,
                                                           rot_mat_local_withDR6[:, index_r])

                        Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                      2) + rot_mat_withDR3.transpose(
                            1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1,
                                                                                                2) + rot_mat_withDR6.transpose(
                            1, 2)
                        orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                        Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                        Dn = torch.matmul(
                            (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                        rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                        Dn_kx = torch.cat(
                            [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                             zeros], dim=1) \
                            .view((batch_size, 3, 3))

                        Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                        DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                        R_derivative[:, indices, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                        rot_mat_local_withDR0[:, -1] = R_derivative[:, indices, parents_6]

                        for index_r in range(1, tree_len):
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local_withDR0[:, index_r])

                        rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])
                        rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local_withDR2[:, index_r])
                        rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local_withDR3[:, index_r])
                        rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local_withDR4[:, index_r])
                        rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local_withDR5[:, index_r])
                        rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local_withDR6[:, index_r])
                        Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                        Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                        Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                        jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_5[:, :, 0]

        # leaf_index = [15, 22, 23, 10, 11]
        # rotate_rest_pose[:, leaf_index] = rotate_rest_pose[:, self.parents[leaf_index]] + torch.matmul(
        #     rot_mat_chain[:, self.parents[leaf_index]],
        #     rel_rest_pose[:, leaf_index]
        # )
        rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        return jacobian.reshape(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mat_local

    def forward_jacobian_and_pred_v3_train(self,
                                           pose_axis_angle,
                                           pose_skeleton,
                                           rest_J,
                                           global_orient,
                                           rot_mat_twist=None,
                                           rotmat_leaf=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)
        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)  ### 这个地方梯度得不到回传！！！！！！！
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ##防止梯度截断
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = [global_orient_mat]
        rot_mat_local = [global_orient_mat]

        jacobian = torch.zeros([batch_size, 18, 24, 3], dtype=torch.float32, device=device)
        R_derivative = torch.zeros((batch_size, 24, 24, 3, 3), dtype=torch.float32,
                                   device=device)  # example:  R_derivative[2,1] == DR2/Dalpha1
        ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
        # index_24_to_18 = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1]  # 受到影响的父节点的数量
        index_24_to_18 = torch.tensor(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -1, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, -1, -1])
        leaf_cnt = 0
        for idx_lev in range(4, 24):
            # indices = self.idx_jacobian[idx_lev]
            indices = idx_lev
            # len_indices = len(indices)
            parents_1 = self.parents[indices]  ## 父节点
            parents_2 = self.parents[parents_1]  ## 父节点的父节点
            parents_3 = self.parents[parents_2]  ## 父节点的父节点的父节点
            parents_4 = self.parents[parents_3]  ## 父节点的父节点的父节点的父节点
            parents_5 = self.parents[parents_4]  ## 父节点的父节点的父节点的父节点
            parents_6 = self.parents[parents_5]  ## 父节点的父节点的父节点的父节点
            parents_7 = self.parents[parents_6]  ## 父节点的父节点的父节点的父节点
            children_1 = self.children_map_opt[indices]

            rotate_rest_pose[:, parents_1] = rotate_rest_pose[:, parents_2] + torch.matmul(
                rot_mat_chain[parents_2],
                rel_rest_pose[:, parents_1]
            )

            orig_vec_unrotate = rel_pose_skeleton[:, indices]
            if idx_lev == 12 or idx_lev == 13 or idx_lev == 14:
                orig_vec_unrotate = 0 * orig_vec_unrotate
            orig_vec = torch.matmul(
                rot_mat_chain[parents_2].transpose(1, 2),
                orig_vec_unrotate
            )
            child_rest_loc = rel_rest_pose[:, indices]  # need rotation back ?
            # (B, K, 3, 1)
            w = torch.cross(child_rest_loc, orig_vec, dim=1)
            w_norm = torch.norm(w, dim=1, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, parents_1].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, parents_1].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)
            # (B, K, 3, 1)
            axis = w / (w_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            # rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            rot_mat_spin = rot_mat_twist[:, parents_1]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)
            # rot_mat_chain[:, parents_1] = torch.matmul(
            #     rot_mat_chain[:, parents_2],
            #     rot_mat)
            # rot_mat_local[:, parents_1] = rot_mat

            if idx_lev != 13 and idx_lev != 14:
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[parents_2],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            q_idex = indices
            q_idex_child = parents_1
            tree_len = len(self.parent_indexs[indices])
            # parent_index = torch.tensor(self.parent_indexs[indices])
            parent_index = self.parent_indexs[indices]

            if children_1 == -1:
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, parents_1] + torch.matmul(
                    rot_mat_chain[parents_1],
                    rel_rest_pose[:, indices]
                )
                # rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                # leaf_cnt += 1
                # # rot_mat_chain[:, indices] = torch.matmul(
                # #     rot_mat_chain[:, parents_1],
                # #     rot_mat)
                # # rot_mat_local[:, indices] = rot_mat
                # rot_mat_chain.append(torch.matmul(
                #     rot_mat_chain[parents_1],
                #     rot_mat))
                # rot_mat_local.append(rot_mat)

            if indices == 12 or indices == 23:
                rot_mat1 = rotmat_leaf[:, leaf_cnt, :, :]
                rot_mat2 = rotmat_leaf[:, leaf_cnt + 1, :, :]
                leaf_cnt += 2
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat1))
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 2]],
                    rot_mat2))
                rot_mat_local.append(rot_mat1)
                rot_mat_local.append(rot_mat2)
            elif indices == 17:
                rot_mat = rotmat_leaf[:, leaf_cnt, :, :]
                leaf_cnt += 1
                rot_mat_chain.append(torch.matmul(
                    rot_mat_chain[self.parents[parents_1 + 1]],
                    rot_mat))
                rot_mat_local.append(rot_mat)

            for x_count in range(tree_len - 1):

                if x_count == tree_len - 2:  ####  dq1/dalpha1  dq2/dalpha2  dq3/dalpha3   dq4/dalpha4  .......
                    DR_sw = cos * K + sin * torch.matmul(K, K)
                    R_derivative[:, parents_1, parents_1] = torch.matmul(DR_sw,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......
                    DR_ = torch.matmul(rot_mat_chain[parents_2], R_derivative[:, parents_1, parents_1])
                    Dq_1 = torch.matmul(DR_, child_rest_loc)
                    jacobian[:, index_24_to_18[parents_1], q_idex] = Dq_1[:, :, 0]

                elif x_count == tree_len - 3:  ####  dq2/dalpha1  dq3/dalpha2  dq4/dalpha3   dq5/dalpha4  .......

                    DR_k_1_k_1 = R_derivative[:, parents_2, parents_2]
                    # rot_mat_local_withDR0 = rot_mat_local[:, parent_index].clone()
                    # rot_mat_local_withDR1 = rot_mat_local_withDR0.clone()
                    # rot_mat_local_withDR1[:, -2] = DR_k_1_k_1
                    # rot_mat_withDR0 = rot_mat_local_withDR0[:, 0]
                    # rot_mat_withDR1 = rot_mat_local_withDR1[:, 0]
                    # for index_r in range(1, tree_len - 1):
                    #     rot_mat_withDR1 = torch.matmul(rot_mat_withDR1,
                    #                                    rot_mat_local_withDR1[:, index_r])

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_1)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn_k = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / ((w_norm + 1e-8) ** 3), Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn_k, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1).view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_2] = torch.matmul(DR_sw_k,
                                                                         rot_mat_spin)  ####  dR1/dalpha1  dR2/dalpha2  dR3/dalpha3   dR4/dalpha4  .......

                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_2]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_2])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    # rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local_withDR1[:, index_r])

                    Dq_k_1_k_1 = jacobian[:, index_24_to_18[parents_2], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1
                    Dq_k_k_1 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_1
                    jacobian[:, index_24_to_18[parents_2], q_idex] = Dq_k_k_1[:, :, 0]

                elif x_count == tree_len - 4:  ####  dq3/dalpha1  dq4/dalpha2  dq5/dalpha3   dq6/dalpha4  .......

                    DR_k_2_k_2 = R_derivative[:, parents_3, parents_3]
                    DR_k_1_k_2 = R_derivative[:, parents_2, parents_3]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_2)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_2)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_3] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_3])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                    Dq_k_1_k_2 = jacobian[:, index_24_to_18[parents_3], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2
                    Dq_k_k_2 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_2
                    jacobian[:, index_24_to_18[parents_3], q_idex] = Dq_k_k_2[:, :, 0]

                elif x_count == tree_len - 5:  ####  dq4/dalpha1  dq5/dalpha2  dq6/dalpha3   dq7/dalpha4  .......

                    DR_k_3_k_3 = R_derivative[:, parents_4, parents_4]
                    DR_k_2_k_3 = R_derivative[:, parents_3, parents_4]
                    DR_k_1_k_3 = R_derivative[:, parents_2, parents_4]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_3)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_3)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_3)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)

                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)

                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_4] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_4]
                    #
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_4])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                    Dq_k_1_k_3 = jacobian[:, index_24_to_18[parents_4], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3
                    Dq_k_k_3 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_3
                    jacobian[:, index_24_to_18[parents_4], q_idex] = Dq_k_k_3[:, :, 0]

                elif x_count == tree_len - 6:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_4_k_4 = R_derivative[:, parents_5, parents_5]
                    DR_k_3_k_4 = R_derivative[:, parents_4, parents_5]
                    DR_k_2_k_4 = R_derivative[:, parents_3, parents_5]
                    DR_k_1_k_4 = R_derivative[:, parents_2, parents_5]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_4)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_4)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_4)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_4)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_5] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_5]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_5])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                    Dq_k_1_k_4 = jacobian[:, index_24_to_18[parents_5], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_4
                    jacobian[:, index_24_to_18[parents_5], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 7:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_5_k_5 = R_derivative[:, parents_6, parents_6]
                    DR_k_4_k_5 = R_derivative[:, parents_5, parents_6]
                    DR_k_3_k_5 = R_derivative[:, parents_4, parents_6]
                    DR_k_2_k_5 = R_derivative[:, parents_3, parents_6]
                    DR_k_1_k_5 = R_derivative[:, parents_2, parents_6]

                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_5)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])

                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_5)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])

                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_5)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])

                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_5)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])

                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_5)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) + rot_mat_withDR5.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))
                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_6] = torch.matmul(DR_sw_k, rot_mat_spin)
                    # rot_mat_local_withDR0[:, -1] = R_derivative[:, parents_1, parents_6]
                    # for index_r in range(1, tree_len):
                    #     rot_mat_withDR0 = torch.matmul(rot_mat_withDR0,
                    #                                    rot_mat_local_withDR0[:, index_r])
                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_6])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])

                    Dq_k_1_k_5 = jacobian[:, index_24_to_18[parents_6], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5
                    Dq_k_k_4 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_5
                    jacobian[:, index_24_to_18[parents_6], q_idex] = Dq_k_k_4[:, :, 0]

                elif x_count == tree_len - 8:  ####  dq5/dalpha1  dq6/dalpha2  dq7/dalpha3   dq8/dalpha4  .......

                    DR_k_6_k_6 = R_derivative[:, parents_7, parents_7]
                    DR_k_5_k_6 = R_derivative[:, parents_6, parents_7]
                    DR_k_4_k_6 = R_derivative[:, parents_5, parents_7]
                    DR_k_3_k_6 = R_derivative[:, parents_4, parents_7]
                    DR_k_2_k_6 = R_derivative[:, parents_3, parents_7]
                    DR_k_1_k_6 = R_derivative[:, parents_2, parents_7]
                    rot_mat_withDR0 = rot_mat_local[0].clone()
                    rot_mat_withDR1 = rot_mat_local[0].clone()
                    rot_mat_withDR2 = rot_mat_local[0].clone()
                    rot_mat_withDR3 = rot_mat_local[0].clone()
                    rot_mat_withDR4 = rot_mat_local[0].clone()
                    rot_mat_withDR5 = rot_mat_local[0].clone()
                    rot_mat_withDR6 = rot_mat_local[0].clone()
                    for index_r in parent_index[1:-1]:
                        if index_r == parent_index[-2]:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, DR_k_1_k_6)
                        else:
                            rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                        if index_r == parent_index[-3]:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, DR_k_2_k_6)
                        else:
                            rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                        if index_r == parent_index[-4]:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, DR_k_3_k_6)
                        else:
                            rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                        if index_r == parent_index[-5]:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, DR_k_4_k_6)
                        else:
                            rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                        if index_r == parent_index[-6]:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, DR_k_5_k_6)
                        else:
                            rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                        if index_r == parent_index[-7]:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, DR_k_6_k_6)
                        else:
                            rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])

                    Temp_derivative = rot_mat_withDR1.transpose(1, 2) + rot_mat_withDR2.transpose(1,
                                                                                                  2) + rot_mat_withDR3.transpose(
                        1, 2) + rot_mat_withDR4.transpose(1, 2) \
                                      + rot_mat_withDR5.transpose(1, 2) + rot_mat_withDR6.transpose(1, 2)
                    orig_vec_inv = torch.matmul(Temp_derivative, orig_vec_unrotate)
                    Dw = torch.cross(child_rest_loc, orig_vec_inv, dim=1)
                    Dn = torch.matmul(
                        (ident / (w_norm + 1e-8)) - torch.matmul(w, w.transpose(1, 2)) / (w_norm + 1e-8) ** 3, Dw)
                    rx_partial, ry_partial, rz_partial = torch.split(Dn, 1, dim=1)
                    Dn_kx = torch.cat(
                        [zeros, -rz_partial, ry_partial, rz_partial, zeros, -rx_partial, -ry_partial, rx_partial,
                         zeros], dim=1) \
                        .view((batch_size, 3, 3))

                    Dn_kxx2 = torch.matmul(Dn_kx, K) + torch.matmul(K, Dn_kx)
                    DR_sw_k = sin * Dn_kx + (1 - cos) * Dn_kxx2
                    R_derivative[:, parents_1, parents_7] = torch.matmul(DR_sw_k, rot_mat_spin)

                    for index_r in parent_index[1:]:
                        if index_r == parent_index[-1]:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, R_derivative[:, parents_1, parents_7])
                        else:
                            rot_mat_withDR0 = torch.matmul(rot_mat_withDR0, rot_mat_local[index_r])

                    rot_mat_withDR1 = torch.matmul(rot_mat_withDR1, rot_mat_local[index_r])
                    rot_mat_withDR2 = torch.matmul(rot_mat_withDR2, rot_mat_local[index_r])
                    rot_mat_withDR3 = torch.matmul(rot_mat_withDR3, rot_mat_local[index_r])
                    rot_mat_withDR4 = torch.matmul(rot_mat_withDR4, rot_mat_local[index_r])
                    rot_mat_withDR5 = torch.matmul(rot_mat_withDR5, rot_mat_local[index_r])
                    rot_mat_withDR6 = torch.matmul(rot_mat_withDR6, rot_mat_local[index_r])
                    Dq_k_1_k_6 = jacobian[:, index_24_to_18[parents_7], q_idex_child].unsqueeze(-1)
                    Temp_q = rot_mat_withDR0 + rot_mat_withDR1 + rot_mat_withDR2 + rot_mat_withDR3 + rot_mat_withDR4 + rot_mat_withDR5 + rot_mat_withDR6
                    Dq_k_k_5 = torch.matmul(Temp_q, child_rest_loc) + Dq_k_1_k_6
                    jacobian[:, index_24_to_18[parents_7], q_idex] = Dq_k_k_5[:, :, 0]

        # rotate_rest_pose = rotate_rest_pose.squeeze(-1).contiguous()
        rotate_rest_pose = rotate_rest_pose.squeeze(-1)
        new_joints = rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return jacobian.view(batch_size, 18, 24 * 3).transpose(1, 2), new_joints, rot_mats

    def forward_full(self,
                     pose_axis_angle,
                     pose_skeleton,
                     betas,
                     phis,
                     global_orient,
                     transl=None,
                     return_verts=True,
                     leaf_thetas=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        # # concate root orientation with thetas
        # if global_orient is not None:
        #     full_pose = torch.cat([global_orient, pose_axis_angle], dim=1)
        # else:
        #     full_pose = pose_axis_angle

        #  joints number we need is 24 - 5 - 1    (five leaf_thetas and one root rotation)

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
        #

        final_pose_skeleton = rotate_rest_pose.clone()
        final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, [0]] + rel_rest_pose[:, [0]]

        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        leaf_thetas = quat_to_rotmat(leaf_thetas)
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 15, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [10, 11, 20, 21],  # 7
            [22, 23],  # 8
            [24, 25, 26, 27, 28]  # 9
        ]

        idx_levs = idx_levs[:-1]

        for idx_lev in range(1, len(idx_levs)):
            indices = idx_levs[idx_lev]
            if idx_lev == len(idx_levs) - 1:
                # leaf nodes
                # rot_mat = leaf_rot_mats[:, :, :, :]
                # parent_indices = [15, 22, 23, 10, 11]
                # rot_mat_local[:, parent_indices] = rot_mat
                # if (torch.det(rot_mat) < 0).any():
                #     print('Something wrong.')
                rot_mat = leaf_rot_mats[:, :, :, :]
                parent_indices = [15, 22, 23, 10, 11]
                rot_mat_local[:, indices] = rot_mat[:, 1:3]
                if (torch.det(rot_mat) < 0).any():
                    print('Something wrong.')
            elif idx_lev == 3:
                # three children
                idx = indices[0]
                # self.children_map_opt[indices] = 12

                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - rotate_rest_pose[:, indices]

                orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
                template_vec = rel_rest_pose[:, self.children_map_opt[indices]]

                norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )

                child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?

                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
                # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices[0]] = rot_mat
            else:
                len_indices = len(indices)
                # (B, K, 3, 1)
                rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rel_rest_pose[:, indices]
                )
                # (B, 3, 1)
                child_final_loc = final_pose_skeleton[:, self.children_map_opt[indices]] - rotate_rest_pose[:, indices]

                orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
                template_vec = rel_rest_pose[:, self.children_map_opt[indices]]

                norm_t = torch.norm(template_vec, dim=2, keepdim=True)  # B x K x 1

                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=2, keepdim=True)  # B x K x 3

                diff = torch.norm(child_final_loc - orig_vec, dim=2, keepdim=True).reshape(-1)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                # child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size * len_indices, 3, 1)
                orig_vec = orig_vec.reshape(batch_size * len_indices, 3, 1)
                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]
                child_final_loc = child_final_loc.reshape(batch_size, len_indices, 3, 1)

                child_final_loc = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                    child_final_loc
                )
                child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
                # (B, K, 1, 1)
                child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
                child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

                # (B, K, 3, 1)
                axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
                axis_norm = torch.norm(axis, dim=2, keepdim=True)

                # (B, K, 1, 1)
                # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
                # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
                cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
                sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

                # (B, K, 3, 1)
                axis = axis / (axis_norm + 1e-8)

                # Convert location revolve to rot_mat by rodrigues
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

                # Convert spin to rot_mat
                # (B, K, 3, 1)
                spin_axis = child_rest_loc / child_rest_norm
                # (B, K, 1, 1)
                rx, ry, rz = torch.split(spin_axis, 1, dim=2)
                zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
                K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                    .view((batch_size, len_indices, 3, 3))
                ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
                # (B, K, 1, 1)
                phi_indices = [item - 1 for item in indices]
                cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
                cos = torch.unsqueeze(cos, dim=3)
                sin = torch.unsqueeze(sin, dim=3)
                rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
                rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

                if (torch.det(rot_mat) < 0).any():
                    print(
                        2,
                        torch.det(rot_mat_loc) < 0,
                        torch.det(rot_mat_spin) < 0
                    )

                rot_mat_chain[:, indices] = torch.matmul(
                    rot_mat_chain[:, self.parents[indices]],
                    rot_mat)
                rot_mat_local[:, indices] = rot_mat

        # (B, K + 1, 3, 3)
        # rot_mats = torch.stack(rot_mat_local, dim=1)
        rot_mats = rot_mat_local

        test_joints = True
        if test_joints:
            J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24],
                                                     dtype=self.dtype)
        else:
            J_transformed = None

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_full_V2(self,
                        pose_axis_angle,
                        pose_skeleton,
                        betas,
                        phis,
                        global_orient,
                        transl=None,
                        return_verts=True,
                        leaf_thetas=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        index_ = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        full_angle[:, index_] = pose_axis_angle[:, :]

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]
        # #
        # final_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)
        # final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, [0]] + rel_rest_pose[:, [0]]

        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        leaf_thetas = quat_to_rotmat(leaf_thetas)
        leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        idx_levs = [
            [0],  # 0
            [3],  # 1
            [6],  # 2
            [9],  # 3
            [1, 2, 12, 13, 14],  # 4
            [4, 5, 15, 16, 17],  # 5
            [7, 8, 18, 19],  # 6
            [10, 11, 20, 21],  # 7
            [22, 23],  # 8
            [24, 25, 26, 27, 28]  # 9
        ]

        idx_levs = idx_levs[:-1]

        # import time
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()

        for idx_lev in range(1, len(idx_levs) - 1):
            indices = idx_levs[idx_lev]
            len_indices = len(indices)
            # (B, K, 3, 1)
            rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rel_rest_pose[:, indices]
            )

            orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            if idx_lev == 3:
                orig_vec = 0 * orig_vec

            orig_vec = torch.matmul(
                rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                orig_vec
            )

            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            # child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            # (B, K, 3, 1)
            axis = torch.cross(child_rest_loc, orig_vec, dim=2)
            axis_norm = torch.norm(axis, dim=2, keepdim=True)

            # (B, K, 1, 1)
            # cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / ( child_rest_norm * child_final_norm + 1e-8)
            # sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)
            cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

            # (B, K, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)

            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]
            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            if (torch.det(rot_mat) < 0).any():
                print(
                    2,
                    torch.det(rot_mat_loc) < 0,
                    torch.det(rot_mat_spin) < 0
                )

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rot_mat)
            rot_mat_local[:, indices] = rot_mat

        indices = idx_levs[-1]
        rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            rot_mat_chain[:, self.parents[indices]],
            rel_rest_pose[:, indices]
        )
        rot_mat = leaf_rot_mats[:, :, :, :]
        # parent_indices = [15, 22, 23, 10, 11]
        rot_mat_local[:, indices] = rot_mat[:, 1:3]

        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # print(elapsed)

        # (B, K + 1, 3, 3)
        rot_mats = rot_mat_local

        test_joints = True
        if test_joints:
            # import time
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24],
                                                     dtype=self.dtype)
            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            # print(elapsed)
        else:
            J_transformed = None

        # new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
        # J_transformed = rotate_rest_pose.squeeze(-1).contiguous()
        # new_joints =rotate_rest_pose - rotate_rest_pose[:, self.root_idx_smpl, :].unsqueeze(1).detach()

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_full_V2_withtwist(self,
                                  pose_axis_angle,
                                  pose_skeleton,
                                  betas,
                                  global_orient,
                                  transl=None,
                                  rotmat_leaf=None,
                                  rot_mat_twist=None):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device
        full_angle = torch.zeros((batch_size, 24), dtype=torch.float32, device=device)

        full_angle[:, self.index_18_to_24] = pose_axis_angle[:, :]

        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        rest_J = vertices2joints(self.J_regressor, v_shaped)

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)

        # rotate the T pose
        rotate_rest_pose = torch.zeros_like(rel_rest_pose)
        # set up the root
        rotate_rest_pose[:, 0] = rel_rest_pose[:, 0]

        # rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1).detach()
        rel_pose_skeleton = torch.unsqueeze(pose_skeleton.clone(), dim=-1)  ## 防止梯度阶段
        rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, self.parents[1:]].clone()
        rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

        global_orient_mat = global_orient

        rot_mat_chain = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local = torch.zeros_like(rot_mat_chain)
        rot_mat_chain[:, 0] = global_orient_mat
        rot_mat_local[:, 0] = global_orient_mat

        # leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        # leaf_thetas = quat_to_rotmat(leaf_thetas)
        # leaf_rot_mats = leaf_thetas.view([batch_size, 5, 3, 3])

        for idx_lev in range(1, len(self.idx_levs) - 1):
            indices = self.idx_levs[idx_lev]
            len_indices = len(indices)
            # (B, K, 3, 1)
            rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rel_rest_pose[:, indices]
            )
            orig_vec = rel_pose_skeleton[:, self.children_map_opt[indices]]
            if idx_lev == 3:
                orig_vec = 0 * orig_vec
            orig_vec = torch.matmul(
                rot_mat_chain[:, self.parents[indices]].transpose(2, 3),
                orig_vec
            )
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 3, 1)
            axis = torch.cross(child_rest_loc, orig_vec, dim=2)
            axis_norm = torch.norm(axis, dim=2, keepdim=True)
            # (B, K, 1, 1)
            cos = full_angle[:, indices].cos().unsqueeze(dim=-1).unsqueeze(dim=-1)
            sin = full_angle[:, indices].sin().unsqueeze(dim=-1).unsqueeze(dim=-1)

            # (B, K, 3, 1)
            axis = axis / (axis_norm + 1e-8)
            # Convert location revolve to rot_mat by rodrigues
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            rot_mat_spin = rot_mat_twist[:, indices]
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            if (torch.det(rot_mat) < 0).any():
                print(
                    2,
                    torch.det(rot_mat_loc) < 0,
                    torch.det(rot_mat_spin) < 0
                )

            rot_mat_chain[:, indices] = torch.matmul(
                rot_mat_chain[:, self.parents[indices]],
                rot_mat)
            rot_mat_local[:, indices] = rot_mat

        indices = self.idx_levs[-1]
        rotate_rest_pose[:, indices] = rotate_rest_pose[:, self.parents[indices]] + torch.matmul(
            rot_mat_chain[:, self.parents[indices]],
            rel_rest_pose[:, indices]
        )

        # rot_mat = leaf_rot_mats[:, :, :, :]
        parent_indices = [15, 22, 23, 10, 11]
        # parent_indices = [22, 23]
        # rot_mat_local[:, parent_indices] = rotmat_leaf[:,1:3]
        rot_mat_local[:, parent_indices] = rotmat_leaf

        # (B, K + 1, 3, 3)
        rot_mats = rot_mat_local

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_full_V3_withtwist(self,
                                  pose_axis_angle,
                                  rest_J,
                                  v_shaped,
                                  transl=None,
                                  rotmat_leaf=None,
                                  rot_mats=None, ):

        batch_size = pose_axis_angle.shape[0]
        device = pose_axis_angle.device

        # parent_indices = [15, 22, 23, 10, 11]
        # parent_indices = [10, 11, 15, 22, 23]
        # rot_mats[:, parent_indices] = rotmat_leaf

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs) \
            .view(batch_size, -1, 3)

        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
            .view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                                   dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_full_V4_withtwist(self,
                                  rest_J,
                                  v_shaped,
                                  transl=None,
                                  rotmat_leaf=None,
                                  rot_mats=None, ):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        # parent_indices = [15, 22, 23, 10, 11]
        # parent_indices = [10, 11, 15, 22, 23]
        # rot_mats[:, parent_indices] = rotmat_leaf

        J_transformed, A = batch_rigid_transform(rot_mats, rest_J[:, :24].clone(), self.parents[:24], dtype=self.dtype)

        # assert torch.mean(torch.abs(rotate_rest_pose - J_transformed)) < 1e-5
        # 4. Add pose blend shapes
        # rot_mats: N x (J + 1) x 3 x 3
        ident = torch.eye(3, dtype=self.dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, self.posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped

        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = self.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=self.dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

        vertices = v_homo[:, :, :3, 0]
        joints_from_verts = vertices2joints(self.J_regressor_h36m, vertices)

        rot_mats = rot_mats.reshape(batch_size * 24, 3, 3)
        rot_mats = rotmat_to_quat(rot_mats).reshape(batch_size, 24 * 4)

        if transl is not None:
            J_transformed += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)
            joints_from_verts += transl.unsqueeze(dim=1)
        else:
            vertices = vertices - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()
            new_joints = J_transformed - J_transformed[:, self.root_idx_smpl, :].unsqueeze(1).detach()
            joints_from_verts = joints_from_verts - joints_from_verts[:, self.root_idx_17, :].unsqueeze(1).detach()

        output = ModelOutput(
            vertices=vertices, joints=new_joints, rot_mats=rot_mats, joints_from_verts=joints_from_verts)
        return output

    def forward_twist(self,
                      rest_J,
                      phis,
                      global_orient):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        global_orient_mat = global_orient
        rot_mat_local = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local[:, 0] = global_orient_mat

        for idx_lev in range(1, len(self.idx_levs) - 1):
            indices = self.idx_levs[idx_lev]
            len_indices = len(indices)
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]
            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_local[:, indices] = rot_mat_spin

        return rot_mat_local

    def forward_twist_and_leaf(self,
                               rest_J,
                               phis,
                               global_orient,
                               leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        global_orient_mat = global_orient
        rot_mat_local = torch.zeros((batch_size, 24, 3, 3), dtype=torch.float32, device=device)
        rot_mat_local[:, 0] = global_orient_mat

        leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        rotmat_leaf = quat_to_rotmat(leaf_thetas)
        rotmat_leaf = rotmat_leaf.view([batch_size, 5, 3, 3])

        for idx_lev in range(1, len(self.idx_levs) - 1):
            indices = self.idx_levs[idx_lev]
            len_indices = len(indices)
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
                .view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 1, 3, 3)
            # (B, K, 1, 1)
            phi_indices = [item - 1 for item in indices]
            cos, sin = torch.split(phis[:, phi_indices], 1, dim=2)
            cos = torch.unsqueeze(cos, dim=3)
            sin = torch.unsqueeze(sin, dim=3)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat_local[:, indices] = rot_mat_spin

        return rot_mat_local, rotmat_leaf

    def forward_twist_and_leaf_train(self,
                                     rest_J,
                                     phis,
                                     global_orient,
                                     leaf_thetas=None):

        batch_size = rest_J.shape[0]
        device = rest_J.device

        rel_rest_pose = rest_J.clone()
        rel_rest_pose[:, 1:] -= rest_J[:, self.parents[1:]].clone()
        rel_rest_pose = torch.unsqueeze(rel_rest_pose, dim=-1)
        phis = phis / (torch.norm(phis, dim=2, keepdim=True) + 1e-8)
        rot_mat_local = [global_orient]

        # leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
        # rotmat_leaf = quat_to_rotmat(leaf_thetas)
        # rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])

        if leaf_thetas is not None:
            leaf_thetas = leaf_thetas.reshape(batch_size * 5, 4)
            rotmat_leaf = quat_to_rotmat(leaf_thetas)
            rotmat_leaf_ = rotmat_leaf.view([batch_size, 5, 3, 3])
        else:
            rotmat_leaf_ = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(batch_size, 5, 1, 1)
            # rotmat_leaf_ = None





        # ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
        # for idx_lev in range(1, len(self.idx_levs)-1):
        for indices in range(1, 24):
            child_rest_loc = rel_rest_pose[:, self.children_map_opt[indices]]  # need rotation back ?
            # (B, K, 1, 1)
            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            # Convert spin to rot_mat
            # (B, K, 3, 1)
            spin_axis = child_rest_loc / child_rest_norm
            # (B, K, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=device)
            K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
            # ident = torch.eye(3, dtype=self.dtype, device=device).reshape(1, 3, 3)
            ident = torch.eye(3, dtype=self.dtype, device=device).unsqueeze(dim=0)

            # (B, K, 1, 1)
            cos, sin = torch.split(phis[:, indices - 1], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)

            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat_local.append(rot_mat_spin)

        # (B, K + 1, 3, 3)
        rot_mats = torch.stack(rot_mat_local, dim=1)

        return rot_mats, rotmat_leaf_

    def single_iteration(self,
                         pose_axis_angle,
                         target,
                         rest_J,
                         phis,
                         global_orient,
                         leaf_thetas=None,
                         rot_mat_twist=None,
                         u=None):

        batch_size = target.shape[0]
        device = target.device

        pred = self.forward_light_V2_withtwist(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            phis=phis,
            global_orient=global_orient,
            leaf_thetas=leaf_thetas,
            rot_mat_twist=rot_mat_twist
        )

        residual = (pred - target).view(batch_size, -1).unsqueeze(-1)

        # mse = np.mean(np.square(residual))
        mse = residual.square().mean(1).squeeze()
        # print(mse)

        jacobian = self.forward_jacobian_v2_batch(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            phis=phis,
            global_orient=global_orient,
            leaf_thetas=leaf_thetas,
            rot_mat_twist=rot_mat_twist
        )

        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)

        ident = torch.eye(18).cuda().reshape(1, 18, 18).repeat(batch_size, 1, 1)

        jtj = jtj + u * ident
        # update = last_mse - mse

        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle - delta), mse

    def single_iteration_v2(self,
                            pose_axis_angle,
                            target,
                            rest_J,
                            global_orient,
                            rot_mat_twist=None,
                            u=None):

        batch_size = target.shape[0]
        device = target.device

        jacobian, pred, rot_mat = self.forward_jacobian_and_pred_v2(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist
        )

        residual = (pred - target).view(batch_size, -1).unsqueeze(-1)
        mse = residual.square().mean(1).squeeze()
        # print(mse)

        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)

        ident = torch.eye(18).cuda().reshape(1, 18, 18).repeat(batch_size, 1, 1)

        jtj = jtj + u * ident
        # update = last_mse - mse

        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle - delta), mse, rot_mat

    def single_iteration_v2_train(self,
                                  pose_axis_angle,
                                  target,
                                  rest_J,
                                  global_orient,
                                  rot_mat_twist=None,
                                  rotmat_leaf=None,
                                  u=None):

        batch_size = target.shape[0]
        device = target.device

        jacobian, pred, rot_mat = self.forward_jacobian_and_pred_v3_train(
            pose_axis_angle=pose_axis_angle,
            pose_skeleton=target,
            rest_J=rest_J,
            global_orient=global_orient,
            rot_mat_twist=rot_mat_twist,
            rotmat_leaf=rotmat_leaf
        )

        residual = (pred - target).reshape(batch_size, 72, 1)
        mse = residual.square().mean(1).squeeze()
        # print(mse)
        jtj = torch.bmm(jacobian.transpose(2, 1), jacobian, out=None)
        ident = torch.eye(18).cuda().reshape(1, 18, 18).repeat(batch_size, 1, 1)
        jtj = jtj + u * ident
        # update = last_mse - mse
        delta = torch.bmm(
            torch.bmm(jtj.inverse(), jacobian.transpose(2, 1)), residual
        ).squeeze()

        return (pose_axis_angle - delta), mse, rot_mat

    def LM_solver(self, target, intrinsic_param, betas=None, joint_root=None, init=None, u=1.0e-2, v=10,
                  pred_phi=None, pred_leaf=None):

        batch_size = target.shape[0]
        device = target.device

        rest_J, v_shaped = self.forward_rest_J(betas)

        global_orient = self.forward_global_orient(pose_skeleton=target, rest_J=rest_J)

        rot_mat_twist, rotmat_leaf = self.forward_twist_and_leaf_train(rest_J=rest_J, phis=pred_phi,
                                                                       global_orient=global_orient,
                                                                       leaf_thetas=pred_leaf)

        params0 = torch.zeros([batch_size, 18], dtype=torch.float32, device=device)
        mse0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
        update0 = torch.zeros([batch_size], dtype=torch.float32, device=device)
        u0 = 1e-2 * torch.ones([batch_size, 18, 18], dtype=torch.float32, device=device)

        params1, mse1, rot_mat1 = self.single_iteration_v2_train(pose_axis_angle=params0, target=target,
                                                                 rest_J=rest_J.clone(),
                                                                 global_orient=global_orient.clone(),
                                                                 rot_mat_twist=rot_mat_twist.clone(),
                                                                 rotmat_leaf=rotmat_leaf.detach().clone(), u=u0)
        update1 = mse0 - mse1
        u_index = (update1 > update0) * (update1 > 0)
        u1 = u0.clone()
        u1[u_index, :, :] /= v
        u1[~u_index, :, :] *= v

        params2, mse2, rot_mat2 = self.single_iteration_v2_train(pose_axis_angle=params1, target=target,
                                                                 rest_J=rest_J.clone(),
                                                                 global_orient=global_orient.clone(),
                                                                 rot_mat_twist=rot_mat_twist.clone(),
                                                                 rotmat_leaf=rotmat_leaf.detach().clone(), u=u1)
        update2 = mse1 - mse2
        u_index = (update2 > update1) * (update2 > 0)
        u2 = u1.clone()
        u2[u_index, :, :] /= v
        u2[~u_index, :, :] *= v

        params3, mse3, rot_mat3 = self.single_iteration_v2_train(pose_axis_angle=params2, target=target,
                                                                 rest_J=rest_J.clone(),
                                                                 global_orient=global_orient.clone(),
                                                                 rot_mat_twist=rot_mat_twist.clone(),
                                                                 rotmat_leaf=rotmat_leaf.detach().clone(), u=u2)
        update3 = mse2 - mse3
        u_index = (update3 > update2) * (update3 > 0)
        u3 = u2.clone()
        u3[u_index, :, :] /= v
        u3[~u_index, :, :] *= v

        params4, mse4, rot_mat4 = self.single_iteration_v2_train(pose_axis_angle=params3, target=target,
                                                                 rest_J=rest_J.clone(),
                                                                 global_orient=global_orient.clone(),
                                                                 rot_mat_twist=rot_mat_twist.clone(),
                                                                 rotmat_leaf=rotmat_leaf.detach().clone(), u=u3)
        update4 = mse3 - mse4
        u_index = (update4 > update3) * (update4 > 0)
        u4 = u3.clone()
        u4[u_index, :, :] /= v
        u4[~u_index, :, :] *= v

        params5, mse5, rot_mat5 = self.single_iteration_v2_train(pose_axis_angle=params4, target=target,
                                                                 rest_J=rest_J.clone(),
                                                                 global_orient=global_orient.clone(),
                                                                 rot_mat_twist=rot_mat_twist.clone(),
                                                                 rotmat_leaf=rotmat_leaf.detach().clone(), u=u4)

        output = self.forward_full_V4_withtwist(
            rest_J=rest_J.clone(),
            v_shaped=v_shaped.clone(),
            rotmat_leaf=rotmat_leaf.clone(),
            rot_mats=rot_mat5
        )



        return output



