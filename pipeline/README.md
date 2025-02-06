## Setup

### Setup environment

1. Install CUDA (only up to cuDNN): https://gist.github.com/MihailCosmin/affa6b1b71b43787e9228c25fe15aeba
2. Add "export CUDA_HOME=/usr/local/cuda/cuda-11.8" and "export CUDA_PATH=$CUDA_HOME" to ~/.zshrc (or whatever .shellrc)
3. Install NerfStudio pre-requisites:
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

### Install project and commands

`pip install -e .`

### Blender renderer

The blender renderer requires some additional setup.
Follow these instructions for downloading blender and starting an x-server: https://github.com/allenai/objaverse-rendering/

The binary is expected to be located in "blender/", but this can also be configured with --pipeline.renderer.blender-binary="..."
