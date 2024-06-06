import bpy
import numpy as np
import json
from mathutils import Matrix, Vector
import os
import argparse
import math
import glob

def add_light_source(camera, offset=Vector((2, 2, 2))):
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    light_data = bpy.data.lights.new(name="NewPointLight", type='POINT')
    light_object = bpy.data.objects.new(name="NewPointLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_object)

    light_data.energy = 2000
    light_data.color = (1, 1, 1)  # 白色

    light_object.location = camera.location + camera.matrix_world.to_quaternion() @ offset


def setup_scene_with_light(light):
    if light:
        camera = bpy.context.scene.camera
        if not camera:
            print("Error: No camera found in the scene.")
            return
        add_light_source(camera, Vector((2, 2, 2)))


def apply_color_to_objects(color=(1.0, 0.0, 0.0, 1.0)):
    for obj in bpy.context.selected_objects:
        mat = bpy.data.materials.new(name="ObjectColor")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = color
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)


def setup_initial_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.location = (-5.49251, 2.86771, 4.40544)
    camera.rotation_euler = (
        math.radians(58.343),
        math.radians(-1.2909),
        math.radians(245.306)
    )
    bpy.context.scene.camera = camera


def setup_scene(obj_path, person_path, color_obj, color_person):
    # 清除所有相机和网格对象
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'CAMERA'}:
            bpy.data.objects.remove(obj, do_unlink=True)


    bpy.ops.import_scene.obj(filepath=obj_path)
    for obj in bpy.context.selected_objects:
        obj.rotation_euler = (0, 0, 0)
    apply_color_to_objects(color_obj)  # 设置导入对象的颜色

    bpy.ops.import_scene.obj(filepath=person_path)
    for obj in bpy.context.selected_objects:
        obj.rotation_euler = (0, 0, 0)
    apply_color_to_objects(color_person)  # 设置导入对象的颜色
    setup_initial_camera()




def add_checkerboard_floor(size=10, tiles=10, color1=(0.786826, 0.751143, 0.8, 1),
                           color2=(0.199955, 0.174482, 0.191498, 1)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'CheckerboardFloor'
    mat = bpy.data.materials.new(name="CheckerboardMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    checker_texture = nodes.new(type='ShaderNodeTexChecker')
    checker_texture.inputs['Scale'].default_value = tiles
    checker_texture.inputs['Color1'].default_value = color1  # 自定义颜色1
    checker_texture.inputs['Color2'].default_value = color2  # 自定义颜色2

    diffuse_shader = nodes.new(type='ShaderNodeBsdfDiffuse')
    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    links.new(checker_texture.outputs['Color'], diffuse_shader.inputs['Color'])
    links.new(diffuse_shader.outputs['BSDF'], material_output.inputs['Surface'])

    if floor.data.materials:
        floor.data.materials[0] = mat
    else:
        floor.data.materials.append(mat)


def render_and_save_image(render_path, image_name):
    if not bpy.context.scene.camera:
        raise Exception("No camera found in scene. Cannot render.")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = os.path.join(render_path, image_name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # 设置输出格式

    bpy.ops.render.render(write_still=True)


def ensure_import_plugin_enabled():
    addon = "io_scene_obj"
    if addon not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=addon)


def main(args):
    color_obj = (0.8, 0.2, 0.2, 1)
    color_person = (0.230057, 0.1923, 0.6, 1)
    # 加载参数，设置场景和相机
    start = int(sorted(glob.glob(os.path.join(args.vis_path, '../object/refine/json/*.json')))[0].split('/')[-1].split('.')[0])
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_adaptive_sampling = True

    ensure_import_plugin_enabled()
    light = True
    for i in range(start, start+args.end):
        obj_path = os.path.join(args.vis_path, 'obj', f"{i}.obj".zfill(10))
        person_path = os.path.join(args.vis_path,'obj', f"{i}_h.obj".zfill(12))
        setup_scene(obj_path, person_path, color_obj, color_person)

        setup_scene_with_light(light)
        light = False
        add_checkerboard_floor(size=20, tiles=20, color1=(0.8, 0.8, 0.8, 1), color2=(0.2, 0.2, 0.2, 1))
        render_path = os.path.join(args.vis_path, "renders")
        os.makedirs(render_path, exist_ok=True)
        image_name = f"rendered_{i:06}.png"
        render_and_save_image(render_path, image_name)



if __name__ == "__main__":
    import sys
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_path', type=str,
                        default='/nas/nas_10/NeuralDome/Hodome/mocap/subject02_desk/vis/')
    parser.add_argument('--end', type=int, default='5')
    args = parser.parse_args(argv)
    main(args)