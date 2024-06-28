import bpy
import numpy as np
import json
from mathutils import Matrix, Vector
import os
import argparse
import math
import glob

# Function to add a light source to the scene
def add_light_source(camera, offset=Vector((2, 2, 2))):
    # Remove existing lights
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    # Create a new point light
    light_data = bpy.data.lights.new(name="NewPointLight", type='POINT')
    light_object = bpy.data.objects.new(name="NewPointLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_object)

    # Set light properties
    light_data.energy = 2000
    light_data.color = (1, 1, 1)  # White light

    # Position the light relative to the camera
    light_object.location = camera.location + camera.matrix_world.to_quaternion() @ offset

# Function to set up the scene with lighting
def setup_scene_with_light(enable_light):
    if enable_light:
        camera = bpy.context.scene.camera
        if not camera:
            print("Error: No camera found in the scene.")
            return
        add_light_source(camera, Vector((2, 2, 2)))

# Function to apply color to selected objects
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

# Function to add a camera to the scene and set its initial position
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

# Function to set up the scene by importing objects and setting their properties
def setup_scene(obj_filepath, person_filepath, obj_color, person_color):
    # Clear all mesh and camera objects from the scene
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'CAMERA'}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Import the object and set its rotation and color
    bpy.ops.import_scene.obj(filepath=obj_filepath)
    for obj in bpy.context.selected_objects:
        obj.rotation_euler = (0, 0, 0)
    apply_color_to_objects(obj_color)

    # Import the person mesh and set its rotation and color
    bpy.ops.import_scene.obj(filepath=person_filepath)
    for obj in bpy.context.selected_objects:
        obj.rotation_euler = (0, 0, 0)
    apply_color_to_objects(person_color)

    # Set up the initial camera position
    setup_initial_camera()

# Function to add a checkerboard floor to the scene
def add_checkerboard_floor(size=10, tiles=10, color1=(0.786826, 0.751143, 0.8, 1),
                           color2=(0.199955, 0.174482, 0.191498, 1)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'CheckerboardFloor'

    # Create a checkerboard material
    mat = bpy.data.materials.new(name="CheckerboardMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Remove all default nodes
    for node in nodes:
        nodes.remove(node)

    # Create and configure checker texture node
    checker_texture = nodes.new(type='ShaderNodeTexChecker')
    checker_texture.inputs['Scale'].default_value = tiles
    checker_texture.inputs['Color1'].default_value = color1
    checker_texture.inputs['Color2'].default_value = color2

    # Create diffuse shader and material output nodes
    diffuse_shader = nodes.new(type='ShaderNodeBsdfDiffuse')
    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    # Link the nodes
    links.new(checker_texture.outputs['Color'], diffuse_shader.inputs['Color'])
    links.new(diffuse_shader.outputs['BSDF'], material_output.inputs['Surface'])

    # Assign the material to the floor
    if floor.data.materials:
        floor.data.materials[0] = mat
    else:
        floor.data.materials.append(mat)

# Function to docs the scene and save the image
def render_and_save_image(output_dir, image_name):
    if not bpy.context.scene.camera:
        raise Exception("No camera found in scene. Cannot docs.")

    # Set docs settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = os.path.join(output_dir, image_name)
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Render the scene and save the image
    bpy.ops.render.render(write_still=True)

# Function to ensure the OBJ import plugin is enabled
def ensure_import_plugin_enabled():
    addon = "io_scene_obj"
    if addon not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=addon)

# Main function to process and docs the scenes
def main(args):
    obj_color = (0.8, 0.2, 0.2, 1)
    person_color = (0.230057, 0.1923, 0.6, 1)

    # Load OBJ file paths
    start_frame = int(sorted(glob.glob(os.path.join(args.obj_path, 'human/*.obj')))[0].split('/')[-1].split('.')[0])
    end_frame = int(sorted(glob.glob(os.path.join(args.obj_path, 'human/*.obj')))[-1].split('/')[-1].split('.')[0])

    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.use_adaptive_sampling = True

    ensure_import_plugin_enabled()

    enable_light = True
    for frame in range(start_frame, end_frame + 1):
        obj_filepath = os.path.join(args.obj_path, 'object', str(frame).zfill(6) + '.obj')
        person_filepath = os.path.join(args.obj_path, 'human', str(frame).zfill(6) + '.obj')

        setup_scene(obj_filepath, person_filepath, obj_color, person_color)
        setup_scene_with_light(enable_light)
        enable_light = False

        add_checkerboard_floor(size=20, tiles=20, color1=(0.8, 0.8, 0.8, 1), color2=(0.2, 0.2, 0.2, 1))

        output_dir = os.path.join(args.vis_path, "blender")
        os.makedirs(output_dir, exist_ok=True)
        image_name = f"{frame:06}.png"
        render_and_save_image(output_dir, image_name)

if __name__ == "__main__":
    import sys
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str,
                        default='/nas/nas_10/NeuralDome/Hodome/mocap/subject02_desk/obj/')
    parser.add_argument('--vis_path', type=str,
                        default='/nas/nas_10/NeuralDome/Hodome/vis/subject02_desk/')
    args = parser.parse_args(argv)
    main(args)
