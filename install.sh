#!/bin/bash
# Install dependency
# 

### New conda environments ###
conda create --name antideepfake python==3.9.0
eval "$(conda shell.bash hook)"
conda activate antideepfake
conda install pip==24.0

### Install PyTorch ###
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

### Install Fariseq ###
# fairseq 0.10.2 on pip does not work
git clone https://github.com/pytorch/fairseq
cd fairseq
# checkout this specific commit. Latest commit does not work
git checkout 862efab86f649c04ea31545ce28d13c59560113d
pip install --editable .
cd ../

### Install SpeechBrain ###
pip install speechbrain==1.0.2

### Install other packages ###
pip install tensorboard tensorboardX soundfile pandarallel scikit-learn numpy==1.21.2 pandas==1.4.3 scipy==1.7.2
