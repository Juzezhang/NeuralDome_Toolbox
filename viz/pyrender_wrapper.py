"""Self-contained pyrender offscreen renderer for the HoDome SMPL-X / object visualization.

Adapted from EasyMocapPublic (easymocap/visualize/pyrender_wrapper.py, author Qing Shuai,
Apache-2.0) — the `get_flags` helper and color tables are inlined here so this module has no
dependency outside `pyrender / numpy / trimesh / opencv`.
"""
import numpy as np
import trimesh
import cv2
import pyrender
from pyrender import RenderFlags

_RENDER_FLAGS_DEFAULT = {
    'flip_wireframe': False, 'all_wireframe': False, 'all_solid': True,
    'shadows': False, 'vertex_normals': False, 'face_normals': False,
    'cull_faces': True, 'point_size': 1.0, 'rgba': True,
}


def get_flags(flags):
    rf = _RENDER_FLAGS_DEFAULT.copy(); rf.update(flags)
    out = RenderFlags.NONE
    if rf['flip_wireframe']:
        out |= RenderFlags.FLIP_WIREFRAME
    elif rf['all_wireframe']:
        out |= RenderFlags.ALL_WIREFRAME
    elif rf['all_solid']:
        out |= RenderFlags.ALL_SOLID
    if rf['shadows']:
        out |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
    if rf['vertex_normals']:
        out |= RenderFlags.VERTEX_NORMALS
    if rf['face_normals']:
        out |= RenderFlags.FACE_NORMALS
    if not rf['cull_faces']:
        out |= RenderFlags.SKIP_CULL_FACES
    if rf['rgba']:
        out |= RenderFlags.RGBA
    return out


def offscree_render(renderer, scene, img, flags):
    rend_rgba, rend_depth = renderer.render(scene, flags=flags)
    assert rend_depth.max() < 65, 'depth should less than 65.536: {}'.format(rend_depth.max())
    rend_depth = (rend_depth * 1000).astype(np.uint16)
    if rend_rgba.shape[2] == 3:  # fail to generate transparent channel
        valid_mask = (rend_depth > 0)[:, :, None]
        rend_rgba = np.dstack((rend_rgba, (valid_mask * 255).astype(np.uint8)))
    rend_rgba = rend_rgba[..., [2, 1, 0, 3]]
    rend_cat = img.copy()
    rend_cat[rend_rgba[:, :, -1] == 255] = rend_rgba[:, :, :3][rend_rgba[:, :, -1] == 255]
    return rend_rgba, rend_depth, rend_cat


class Renderer:
    def __init__(self, bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5], flags={}) -> None:
        self.bg_color = bg_color
        self.ambient_light = ambient_light
        self.renderer = pyrender.OffscreenRenderer(1024, 1024)
        self.flags = get_flags(flags)

    @staticmethod
    def add_light(scene, camera=None):
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3)
        scene.add(light, pose=np.eye(4))
        light_z = np.eye(4)
        light_z[:3, :3] = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        scene.add(light, pose=light_z)

    def __call__(self, render_data, images, cameras, extra_mesh=[],
                 ret_image=False, ret_depth=False, ret_color=False, ret_mask=False, ret_all=True):
        if isinstance(images, np.ndarray) and isinstance(cameras, dict):
            images, cameras = [images], [cameras]
        assert isinstance(cameras, list)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        output_images, output_colors, output_depths = [], [], []
        for nv, img in enumerate(images):
            cam = cameras[nv]
            K, R, T = cam['K'], cam['R'], cam['T']
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            scene = pyrender.Scene(bg_color=self.bg_color, ambient_light=self.ambient_light)
            for iextra, _mesh in enumerate(extra_mesh):
                mesh = _mesh.copy()
                trans_cam = np.eye(4); trans_cam[:3, :3] = R; trans_cam[:3, 3:] = T
                mesh.apply_transform(trans_cam); mesh.apply_transform(rot)
                scene.add(pyrender.Mesh.from_trimesh(mesh), 'extra{}'.format(iextra))
            for trackId, data in render_data.items():
                vert = data['vertices'].copy() @ R.T + T.T
                faces = data['faces']
                if 'colors' not in data.keys():
                    col = get_colors(data.get('vid', trackId))
                    mesh = trimesh.Trimesh(vert, faces, process=False)
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0, roughnessFactor=0.0, alphaMode='OPAQUE', baseColorFactor=col)
                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=data.get('smooth', True))
                else:
                    mesh = trimesh.Trimesh(vert, faces, vertex_colors=data['colors'], process=False)
                    mesh.apply_transform(rot)
                    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(mesh, data['name'])
            camera = pyrender.camera.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
            scene.add(camera, pose=np.eye(4))
            self.add_light(scene, camera=cam)
            rend_rgba, rend_depth, rend_cat = offscree_render(self.renderer, scene, img, self.flags)
            output_colors.append(rend_rgba); output_depths.append(rend_depth); output_images.append(rend_cat)
        if ret_depth:
            return output_depths
        if ret_color:
            return output_colors
        if ret_mask:
            return [val[:, :, 3] for val in output_colors]
        if ret_image:
            return output_images
        return output_colors, output_depths, output_images

    def render_image(self, render_data, images, cameras, extra_mesh, **kwargs):
        return self.__call__(render_data, images, cameras, extra_mesh, ret_all=True, **kwargs)


# BGR-ordered (composited onto BGR images)
colors = [
    (94 / 255, 124 / 255, 226 / 255), (255 / 255, 200 / 255, 87 / 255), (74 / 255., 189 / 255., 172 / 255.),
    (8 / 255, 76 / 255, 97 / 255), (219 / 255, 58 / 255, 52 / 255), (77 / 255, 40 / 255, 49 / 255),
]
colors_table = {
    '_blue': [0.65098039, 0.74117647, 0.85882353], '_pink': [.9, .7, .7],
    '_mint': [166 / 255., 229 / 255., 204 / 255.], '_green': [153 / 255., 216 / 255., 201 / 255.],
    '_red': [251 / 255., 128 / 255., 114 / 255.], '_orange': [253 / 255., 174 / 255., 97 / 255.],
    '_yellow': [250 / 255., 230 / 255., 154 / 255.],
    'r': [1, 0, 0], 'g': [0, 1, 0], 'b': [0, 0, 1], 'k': [0, 0, 0], 'y': [1, 1, 0], 'purple': [128 / 255, 0, 128 / 255],
}


def get_colors(pid):
    if isinstance(pid, int):
        return colors[pid % len(colors)]
    elif isinstance(pid, str):
        return colors_table[pid]
    elif isinstance(pid, (list, tuple)):
        if len(pid) == 3:
            pid = (pid[0], pid[1], pid[2], 1.)
        assert len(pid) == 4
        return pid
