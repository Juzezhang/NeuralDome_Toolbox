# NeuralDome Dataset Toolbox

[//]: # ([![PWC]&#40;https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hybrik-a-hybrid-analytical-neural-inverse/3d-human-pose-estimation-on-3dpw&#41;]&#40;https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=hybrik-a-hybrid-analytical-neural-inverse&#41;)

<div align="center">
<img src="assets/NeuralDome.png">
</div>


This repository has a toolbox to download, process and visualize the [`NeuralDome Dataset`](https://drive.google.com/drive/folders/1-QHvcwa71Wk7rdfnQrOyInqK-SWK6lRA) for the paper:

**NeuralDome: A Neural Modeling Pipeline on Multi-View Human-Object Interactions**

[[`Paper`](https://arxiv.org/pdf/2212.07626.pdf)]
[[`Project Page`](https://juzezhang.github.io/NeuralDome/)]

## Note
Jan. 05, 2024: We are currently in the process of uploading the whole dataset to Google Cloud Drive. Since the file size exceeds 5TB, it may take more than a few weeks to complete the upload.

## Environment

``` bash
# 1. Create a conda virtual environment.
conda create -n NeuralDome python=3.8 pytorch=1.11 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate NeuralDome

# 2. Install PyTorch3D-0.4.0
git clone --branch v0.4.0 https://github.com/facebookresearch/pytorch3d.git 
(otherwise, you can also download the 0.4.0 version of pytord3d through this [link](https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.zip))
cd pytorch3d
python setup.py install

# 4. Install requirements
pip install -r requirements.txt
```

## Data Preparation
The full dataset contains 76-view RGB videos paired with homask, scanned object template, mocap and geometry. 
Please download the data from this link and extract the tar file.
``` bash
for file in *.tar; do tar -xf "$file"; done
```

## Data Structure
```
├─ HODome
    ├─ images
        ├─ Seq_Name
            ├─ 0
                ├─ 000000.jpg
                ├─ 000001.jpg
                ├─ 000003.jpg
                    ...
            ...
    ├─ videos
        ├─ Seq_Name
            ├─ data1.mp4
            ├─ data2.mp4
            ...
            ├─ data76.mp4
    ├─ mocap
        ├─ Seq_Name
            ├─ keypoints2d
            ├─ keypoints3d
            ├─ object
            ├─ smpl
    ├─ mask
        ├─ Seq_Name
            ├─ hmask
            ├─ omask
├─ dataset_information.json
├─ startframe.json
    ...
```


## Image Extraction
Since the image files are extremely large, we have not uploaded them. Please run the following scripts to extract the image files from the provided videos.
``` bash
python ./scripts/video2image.py
```
## Visualization

To do


## Citing
If our code helps your research, please consider citing the following paper:

        @inproceedings{
              zhang2023neuraldome,
              title={NeuralDome: A Neural Modeling Pipeline on Multi-View Human-Object Interactions},
              author={Juze Zhang and Haimin Luo and Hongdi Yang and Xinru Xu and Qianyang Wu and Ye Shi and Jingyi Yu and Lan Xu and Jingya Wang},
              booktitle={CVPR},
              year={2023},
        }
      