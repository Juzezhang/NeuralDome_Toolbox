# ![Blender](./Blender_logo_no_text.png)Rendering HODome with Blender
## Instructions

This is a script for visualizing the HOdome dataset using Blender.

## 🚀Update
- 2024/06/06: Upload the code


## 📖Prerequisite
**Download & Install:**
- [Blender 3.6](https://mirror.freedif.org/blender/release/Blender3.6/blender-3.6.12-linux-x64.tar.xz)
-  [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

**Body model:**
- Clone the body model code from [EgoEgo](https://github.com/lijiaman/egoego_release/tree/main/body_model).
- Download [SMPL-H](https://mano.is.tue.mpg.de/login.php).

Overall, the structure of `body_model/` folder should be:
```
|-- __init__.py
|-- body_model.py
|-- smplh
|   |-- LICENSE.txt
|   |-- female
|   |   |-- model.npz
|   |-- info.txt
|   |-- male
|   |   |-- model.npz
|   |-- neutral
|       |-- model.npz
|-- utils.py
```

**Structure:**
```
./render/
|-- body_model
|-- small_data
|   |-- scaned_object           # Scaned Object 
|   |-- subject02_desk          # Need visualized sequences
|       |-- object              # Object JSON
|       |-- smplh               # Person JSON
|       |-- vis                 # Visualization
|           |-- obj             # Save Object & Person.obj
|           |-- renders         # Render image
```

## 🥳Quick Start

**Save the SMPL and object parameters as an OBJ file**

```bash
python json2obj.py --file_root './small_data' --object_root './small_data/scaned_object/' --seq 'subject02_desk'
```

**Render the corresponding OBJ file**
```bash
PATH/WHERE/INSTALL/BLENDER/blender -P render_hodome.py -b -- --vis_path './small_data/subject02_desk/vis/'
```





## 🫶Acknowledgements
The script is based on [Blender 3.6](https://mirror.freedif.org/blender/release/Blender3.6/blender-3.6.12-linux-x64.tar.xz) and [Egoego](https://github.com/lijiaman/egoego_release). Thanks for the authors for their efforts.
