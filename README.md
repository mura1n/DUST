This is code and regeneration dataset for paper "One World, Dual Timeline: Decoupled Spatio-Temporal Gaussian Scene Graphs for 4D Cooperative Driving Reconstruction"

# Run DUST code

## Installation

Run the following commands to set up the environment:

```shell
# Clone the repository with submodules
git clone --recursive https://github.com/mura1n/DUST.git
cd DUST

# Create the environment
conda create -n DUST python=3.9 -y
conda activate DUST
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast
```

## Prepare Data
The optimized and regenerated annotaion files have been stored in google driver. All related operations are fully anonymized. A Croissant file is included in the directory to describe the original data and dataset resources. Upon downloading, a ready-to-use data folder is obtained. No additional data preprocessing operations are required. (https://drive.google.com/drive/folders/1Hgpo-zT9P0ZRCbEGn0k-HNhMYJuQhvZV?usp=sharing)

## Running
### Training
```shell
DRIVESTUDIO_CPU_THREADS= [your threads limitation] CUDA_VISIBLE_DEVICES= [your gpu id] python tools/train.py
   --config_file configs/DUST.yaml
   --output_root $output_root
   --project $project
   --run_name $expname
   dataset=kitti/2cams
   data.scene_idx=$sequence id
   data.pixel_source.reference_camera=$choose a reference camera 0/1 as anchor time$
   data.pixel_source.reference_pose_group=roadside
   data.pixel_source.timestamp_paths.cam0=$your real time timestamp .txt for cam0$
   data.pixel_source.timestamp_paths.cam1=$your real time timestamp .txt for cam1$
```

### Evaluation
```shell
python tools/eval.py --resume_from $ckpt_path
```
