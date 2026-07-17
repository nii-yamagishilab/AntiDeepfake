#!/bin/bash
# Install dependency using HuggingFace Transformers backend (no fairseq required).
# Works with Python 3.9-3.14+.
#
# This is an alternative to install.sh that replaces the fairseq dependency
# with HuggingFace transformers, enabling use on modern Python versions.

### New conda environments ###
conda create --name antideepfake python==3.12
eval "$(conda shell.bash hook)"
conda activate antideepfake

### Install PyTorch ###
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

### Install HuggingFace Transformers (replaces fairseq) ###
pip install transformers safetensors

### Install SpeechBrain ###
pip install speechbrain==1.0.2

### Install other packages ###
pip install tensorboard tensorboardX soundfile pandarallel scikit-learn numpy pandas scipy
