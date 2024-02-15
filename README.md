# NeuralDome Dataset Toolbox

<div align="center">
    <img src="assets/NeuralDome.png" alt="NeuralDome Logo">
</div>

Welcome to the repository for the NeuralDome Dataset Toolbox, which facilitates downloading, processing, and visualizing the NeuralDome Dataset. This toolbox supports our publication:

**NeuralDome: A Neural Modeling Pipeline on Multi-View Human-Object Interactions**

[[`Paper`](https://arxiv.org/pdf/2212.07626.pdf)]
[[`Project Page`](https://juzezhang.github.io/NeuralDome/)]
[[`Data`](https://drive.google.com/drive/folders/1-QHvcwa71Wk7rdfnQrOyInqK-SWK6lRA?usp=sharing)]

## Updates
- **Jan. 05, 2024**: Currently uploading the entire dataset to Google Cloud Drive. Due to its size exceeding 5TB, this may take several weeks.
- **Jan. 30, 2024**: Upload of raw video data is now complete!
- **Feb. 9, 2024**: Upload of masks is now complete!

## Setting Up Your Environment

To get started, set up your environment as follows:

```bash
# Create a conda virtual environment
conda create -n NeuralDome python=3.8 pytorch=1.11 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate NeuralDome

# Install PyTorch3D-0.4.0
git clone --branch v0.4.0 https://github.com/facebookresearch/pytorch3d.git 
# Alternatively, download PyTorch3D v0.4.0 directly via this link:
# https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.zip
cd pytorch3d
python setup.py install

# Install other requirements
pip install -r requirements.txt
```

## Preparing the Data

The complete dataset features 76-view RGB videos along with corresponding masks, mocap data, geometry, and scanned object templates. Download and extract the dataset from [this link](https://drive.google.com/drive/folders/1-QHvcwa71Wk7rdfnQrOyInqK-SWK6lRA):

```bash
for file in *.tar; do tar -xf "$file"; done
```

## Data Structure Overview

The dataset is organized as follows:
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
            ├─ homask
            ├─ hmask
            ├─ omask
    ├─ calibration
        ├─ 20221018
        ...
    ├─ dataset_information.json
    ├─ startframe.json
    ...
```


## Extracting Images from Videos

Since the image files are extremely large, we have not uploaded them. Please run the following scripts to extract the image files from the provided videos.

```bash
python ./scripts/video2image.py
```

## Visualization Toolkit

Our `hodome_visualization.py` script showcases how to access the diverse annotations in our dataset. It uses the following command-line arguments:

- `--root_path`: Directory containing the dataset.
- `--seq_name`: Sequence name to process.
- `--resolution`: Output image resolution.
- `--output_path`: Where to save rendered images.

Ensure your environment and data are properly set up before executing the script. Here's an example command:

```bash
python ./scripts/hodome_visualization.py --root_path "/path/to/your/data" --seq_name "subject01_baseball" --resolution 720 --output_path "/path/to/your/output"
```

## Citing Our Work

If you find our toolbox or dataset useful for your research, please consider citing our paper:

```bibtex
@inproceedings{
      zhang2023neuraldome,
      title={NeuralDome: A Neural Modeling Pipeline on Multi-View Human-Object Interactions},
      author={Juze Zhang and Haimin Luo and Hongdi Yang and Xinru Xu and Qianyang Wu and Ye Shi and Jingyi Yu and Lan Xu and Jingya Wang},
      booktitle={CVPR},
      year={2023},
}
```
