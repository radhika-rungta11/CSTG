#!/bin/bash


conda create -n dynamic_c3dgs python=3.10
conda activate dynamic_c3dgs

# PyTorch 2.7.1+cu128: first version with Blackwell (sm_120) support; requires Python 3.9+
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128


# Install for Gaussian Rasterization (Ch9) - Ours-Full
pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9


# install simpleknn
pip install thirdparty/gaussian_splatting/submodules/simple-knn

# install opencv-python-headless, to work with colmap on server
pip install opencv-python
# Install MMCV for CUDA KNN, used for init point sampling, reduce number of points when sfm points are too many
cd thirdparty
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -e .
cd ../../

# other packages
pip install scipy
pip install kornia
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
pip install dahuffman==0.4.1
pip install vector-quantize-pytorch==1.8.1
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install plyfile
pip install scikit-image
conda install colmap -c conda-forge
