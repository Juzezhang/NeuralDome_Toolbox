# ![Blender](./Blender_logo_no_text.png)Rendering HODome with Blender
## Instructions

This script is designed for visualizing the HOdome dataset using Blender.

## 🚀Update
- 2024/06/06: Upload the code


## 📖Prerequisite
**Download & Install:**
- [Blender 3.6](https://mirror.freedif.org/blender/release/Blender3.6/blender-3.6.12-linux-x64.tar.xz)
-  [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

## 🥳Quick Start

**Save the SMPL and object parameters as an OBJ file**

```bash
python scripts/hodome_json2obj.py --root_path "/path/to/your/data" --seq_name "subject01_baseball"
```
By default, the OBJ files will be saved at $root$/mocap/subject01_baseball/obj/.

**Render the corresponding OBJ file**
```bash
PATH/WHERE/INSTALL/BLENDER/blender -P scripts/hodome_visualization_blender.py -b -- --obj_path /nas/nas_10/NeuralDome/Hodome/mocap/subject01_baseball/obj/ --vis_path /nas/nas_10/NeuralDome/Hodome/vis/subject01_baseball/
```


## 🫥TODO


## 🫶Acknowledgements
The script is based on [Blender 3.6](https://mirror.freedif.org/blender/release/Blender3.6/blender-3.6.12-linux-x64.tar.xz) and [Egoego](https://github.com/lijiaman/egoego_release). Thanks for the authors for their efforts.
