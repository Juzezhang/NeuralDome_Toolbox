import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from pytorch3d.io import load_obj , save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import numpy as np
from pytorch3d.transforms import Rotate, Transform3d, Translate
from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast

def rot6d_to_matrix(rot_6d):
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(rotmat):
    return rotmat.view(-1, 3, 3)[:, :, :2]


def make_rotate(rx, ry, rz, angle=True):
    if angle:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    sinX, sinY, sinZ = np.sin(rx), np.sin(ry), np.sin(rz)
    cosX, cosY, cosZ = np.cos(rx), np.cos(ry), np.cos(rz)

    Rx, Ry, Rz = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    Rx[0, 0] = 1.0;
    Rx[1, 1] = cosX;
    Rx[1, 2] = -sinX;
    Rx[2, 1] = sinX;
    Rx[2, 2] = cosX
    Ry[0, 0] = cosY;
    Ry[0, 2] = sinY;
    Ry[1, 1] = 1.0;
    Ry[2, 0] = -sinY;
    Ry[2, 2] = cosY
    Rz[0, 0] = cosZ;
    Rz[0, 1] = -sinZ;
    Rz[1, 0] = sinZ;
    Rz[1, 1] = cosZ;
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def points2mask(v3d, K, E, ref=None):
    if ref is None:
        h = 1280
        w = 720
    else:
        h, w = ref.shape[:2]
    # R = E[:3, :3].T
    # t = -R.dot(E[:3, 3:])
    R = E[:3, :3]
    t = E[:3,3:].reshape(3)

    mask = np.zeros((h, w), 'uint8')
    #     K = intrinc_list[0]
    v2d = K.dot(np.dot(R, v3d.T) + t.reshape(3, 1))
    v2d = v2d / v2d[2:]
    v2d = np.round(v2d).astype('int')
    uv1_masked = v2d.T[np.array(list((v2d[0] > 0)) and list(v2d[0] < w) and list(v2d[1] > 0) and list(v2d[1] < h))].T
    uv1_masked = v2d.T[(v2d[0] < w) & (v2d[0] > 0) & (v2d[1] < h) & (v2d[1] > 0)].T
    (v2d[0] < w) & (v2d[0] > 0)
    mask[uv1_masked[1], uv1_masked[0]] = 255
    return mask


def points_projection(v3d, K, R, t, ref=None):
    if ref is None:
        h = 540
        w = 960
    else:
        h, w = ref.shape[:2]
    # R = E[:3, :3].T
    # t = -R.dot(E[:3, 3:])
    # R = E[:3, :3]
    # t = E[:3,3:].reshape(3)

    mask = np.zeros((h, w), 'uint8')
    #     K = intrinc_list[0]
    v2d = K.dot(np.dot(R, v3d.T) + t.reshape(3, 1))
    v2d = v2d / v2d[2:]
    v2d = np.round(v2d).astype('int')
    uv1_masked = v2d.T[np.array(list((v2d[0] > 0)) and list(v2d[0] < w) and list(v2d[1] > 0) and list(v2d[1] < h))].T
    uv1_masked = v2d.T[(v2d[0] < w) & (v2d[0] > 0) & (v2d[1] < h) & (v2d[1] > 0)].T
    (v2d[0] < w) & (v2d[0] > 0)
    mask[uv1_masked[1], uv1_masked[0]] = 255
    return mask

def image_grid(images, rows=None, cols=None,
               dpi=100, fill: bool = True, show_axes: bool = False, rgb: bool = True,
               ):
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1
    plt.rcParams['figure.dpi'] = dpi
    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[...])
        if not show_axes:
            ax.set_axis_off()
    fig.show()

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def aa_to_quat_numpy(axis_angle):
    """Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a np.ndarray of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as np.ndarray of shape (..., 4).
    """
    angles = np.linalg.norm(axis_angle, ord=2, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        axis=-1)
    return quaternions


def quatern2rotMat_numpy(qua):
    # get the rotation matrix from quatern
    # temp = [qua[3], qua[0], qua[1], qua[2]]
    # matrix = p.getMatrixFromQuaternion(qua)
    # matrix = np.asarray(matrix)
    # matrix = matrix.reshape(3, 3)
    R = np.zeros([3, 3])
    w = qua[0]
    x = qua[1]
    y = qua[2]
    z = qua[3]
    R[0, 0] = w**2 + x**2 - y**2 - z**2
    R[0, 1] = 2 * (x * y + z * w)
    R[0, 2] = 2 * (x * z - y * w)
    R[1, 0] = 2 * (x * y - z * w)
    R[1, 1] = w**2 + y**2 - x**2 - z**2
    R[1, 2] = 2 * (y * z + x * w)
    R[2, 0] = 2 * (x * z + y * w)
    R[2, 1] = 2 * (y * z - x * w)
    R[2, 2] = w**2 + z**2 - x**2 - y**2
    return R


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))



def axis_angle_to_matrix_numpy(axis_angle):
    return quatern2rotMat_numpy(aa_to_quat_numpy(axis_angle))


# Default values for rotation and translation matrices.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)

class OpenGLRealPerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices using the OpenGL convention for a perspective camera.
    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> screen.
    The `transform_points` method calculates the full world -> screen transform
    and then applies it to the input points.
    The transforms can also be returned separately as Transform3d objects.
    """

    def __init__(
            self,
            focal_length=1.0,
            principal_point=((0.0, 0.0),),
            R=r,
            T=t,
            znear=0.01,
            zfar=100.0,
            x0=0,
            y0=0,
            w=640,
            h=480,
            device="cpu",
    ):
        """
        __init__(self, znear, zfar, R, T, device) -> None  # noqa
        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            znear=znear,
            zfar=zfar,
            x0=x0,
            y0=y0,
            h=h,
            w=w,
        )

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the OpenGL perpective projection matrix with a symmetric
        viewing frustrum. Use column major order.
        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.
        Return:
            P: a Transform3d object which represents a batch of projection
            matrices of shape (N, 3, 3)
        .. code-block:: python
            q = -(far + near)/(far - near)
            qn = -2*far*near/(far-near)
            P.T = [
                    [2*fx/w,     0,           0,  0],
                    [0,          -2*fy/h,     0,  0],
                    [(2*px-w)/w, (-2*py+h)/h, -q, 1],
                    [0,          0,           qn, 0],
                ]
                sometimes P[2,:] *= -1, P[1, :] *= -1
        """
        znear = kwargs.get("znear", self.znear)  # pyre-ignore[16]
        zfar = kwargs.get("zfar", self.zfar)  # pyre-ignore[16]
        x0 = kwargs.get("x0", self.x0)  # pyre-ignore[16]
        y0 = kwargs.get("y0", self.y0)  # pyre-ignore[16]
        w = kwargs.get("w", self.w)  # pyre-ignore[16]
        h = kwargs.get("h", self.h)  # pyre-ignore[16]
        principal_point = kwargs.get("principal_point", self.principal_point)  # pyre-ignore[16]
        focal_length = kwargs.get("focal_length", self.focal_length)  # pyre-ignore[16]

        if not torch.is_tensor(focal_length):
            focal_length = torch.tensor(focal_length, device=self.device)

        if len(focal_length.shape) in (0, 1) or focal_length.shape[1] == 1:
            fx = fy = focal_length
        else:
            fx, fy = focal_length.unbind(1)

        if not torch.is_tensor(principal_point):
            principal_point = torch.tensor(principal_point, device=self.device)
        px, py = principal_point.unbind(1)

        P = torch.zeros((self._N, 4, 4), device=self.device, dtype=torch.float32)
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0
        # define P.T directly
        #         P[:, 0, 0] = -2.0 * fx / w  # NOTE: flip x
        #         P[:, 1, 1] = -2.0 * fy / h
        #         P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        #         P[:, 0, 2] = -(-2 * px + w + 2 * x0) / w  # NOTE: x shift
        #         P[:, 1, 2] = -(+2 * py - h + 2 * y0) / h
        #         P[:, 3, 2] = z_sign * ones
        #         P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        P[:, 0, 0] = -2.0 * fx / w  # NOTE: flip x
        P[:, 1, 1] = -2.0 * fy / h
        P[:, 2, 2] = z_sign * (zfar + znear) / (zfar - znear)
        P[:, 2, 0] = (-2 * px + w + 2 * x0) / w  # NOTE: x shift
        P[:, 2, 1] = -(+2 * py - h + 2 * y0) / h
        P[:, 2, 3] = z_sign * ones
        P[:, 3, 2] = - 2 * (zfar * znear) / (zfar - znear)

        #         z_sign = 1.0
        #         # define P.T directly
        #         P[:, 0, 0] = -2.0 * fx / w
        #         P[:, 1, 1] = -2.0 * fy / h
        #         P[:, 2, 0] = -(-2 * px + w + 2 * x0) / w
        #         P[:, 2, 1] = -(+2 * py - h + 2 * y0) / h
        #         P[:, 2, 3] = z_sign * ones

        # NOTE: This part of the matrix is for z renormalization in OpenGL
        # which maps the z to [-1, 1]. This won't work yet as the torch3d
        # rasterizer ignores faces which have z < 0.
        # P[:, 2, 2] = z_sign * (far + near) / (far - near)
        # P[:, 2, 3] = -2.0 * far * near / (far - near)
        # P[:, 2, 3] = z_sign * torch.ones((N))

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane. This replaces the OpenGL z normalization to [-1, 1]
        # until rasterization is changed to clip at z = -1.
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 3, 2] = -(zfar * znear) / (zfar - znear)

        # OpenGL uses column vectors so need to transpose the projection matrix
        # as torch3d uses row vectors.
        transform = Transform3d(device=self.device)
        transform._matrix = P
        return transform

    def clone(self):
        other = OpenGLRealPerspectiveCameras(device=self.device)
        return super().clone(other)

    def get_camera_center(self, **kwargs):
        """
        Return the 3D location of the camera optical center
        in the world coordinates.
        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.
        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.
        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.
        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.
        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R = self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        T = self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        if T.shape[0] != R.shape[0]:
            msg = "Expected R, T to have the same batch dimension; got %r, %r"
            raise ValueError(msg % (R.shape[0], T.shape[0]))
        if T.dim() != 2 or T.shape[1:] != (3,):
            msg = "Expected T to have shape (N, 3); got %r"
            raise ValueError(msg % repr(T.shape))
        if R.dim() != 3 or R.shape[1:] != (3, 3):
            msg = "Expected R to have shape (N, 3, 3); got %r"
            raise ValueError(msg % R.shape)

        # Create a Transform3d object
        T = Translate(T, device=T.device)
        R = Rotate(R, device=R.device)
        world_to_view_transform = R.compose(T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-screen transform composing the
        world-to-view and view-to-screen transforms.
        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.
        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.
        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]

        world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
        view_to_screen_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_screen_transform)

    def transform_points(self, points, **kwargs) -> torch.Tensor:
        """
        Transform input points from world to screen space.
        Args:
            points: torch tensor of shape (..., 3).
        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_screen_transform = self.get_full_projection_transform(**kwargs)
        return world_to_screen_transform.transform_points(points)


    def is_perspective(self):
        return True

    def get_znear(self):
        return self.znear if hasattr(self, "znear") else None

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret



def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
