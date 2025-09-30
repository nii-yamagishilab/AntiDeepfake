<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/logo.png?raw=true" width="60%" alt="AntiDeepfake" />
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NII Yamagishi Lab-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/LICENSE-CODE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-BSD 3-f5de53?&color=f5de53"/></a>
  <a href="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/LICENSE-CHECKPOINT"><img alt="Model License"
    src="https://img.shields.io/badge/Checkpoint_License-CC BY%20NC%20SA 4.0-f5de53?&color=f5de53"/></a>
<a href="https://zenodo.org/records/15580543">
  <img alt="Zenodo" src="https://img.shields.io/badge/%20Zenodo-Checkpoints-0077C8?logo=zenodo&logoColor=white" />
</a>

  <br>
  <a href="https://arxiv.org/abs/2506.21090"><b>Paper Link</b></a>
</div>

<hr>

## 📢 News and Updates

[Sep. 20, 2025] Add functions and a demo notebook for drift detection. Please check details [README_drift.md](./README_drift.md).

[July 4, 2025] We added supplementary information to our training data to help guide your selection of which data to use. Please check [here](./protocols/README.md).

[June 27, 2025] Initial release!!!


## Introduction

The AntiDeepfake project provides a series of powerful foundation models post-trained for deepfake detection. The AntiDeepfake model can be used for feature extraction for deepfake detection in a zero-shot manner, or it may be further fine-tuned and optimized for a specific database or deepfake-related task.

The table below summarizes the Equal Error Rate (EER) performance across multiple evaluation datasets, along with model sizes, to help guide your selection.

For more technical details and analysis, please refer to our paper [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090).

| 🤗 Model                                                                                 | Params | RawBoost | ADD2023 | DEEP-VOICE | FakeOrReal | FakeOrReal-Norm | In-the-Wild | Deepfake-Eval-2024 |
|------------------------------------------------------------------------------------------|--------|----|---------|-----------|------------|--------------|----------|----------|
| [HuBERT-XL-NDA](https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake-nda) | 964M   | ✗  | 35.34   | 14.87     | 3.67       | 15.52        | 17.99    | 47.72    |
| [W2V-Small-NDA](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake-nda) | 95M    | ✗  | 19.41   | 16.22     | 1.05       | 6.47         | 4.65     | 31.97    |
| [W2V-Large-NDA](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake-nda) | 317M   | ✗  | 12.67   | 5.01      | 0.80       | 1.44         | 2.25     | 30.05    |
| [MMS-300M-NDA](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake-nda)       | 317M   | ✗  | 11.22   | 3.04      | 0.46       | 2.71         | 2.00     | 31.38    |
| [MMS-1B-NDA](https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake-nda)           | 965M   | ✗  | 9.46    | 2.27      | 0.89       | 1.10         | 1.86     | 27.55    |
| [XLS-R-1B-NDA](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake-nda)       | 965M   | ✗  | 6.58    | 2.96      | 3.16       | 10.91        | 1.36     | 26.17    |
| [XLS-R-2B-NDA](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake-nda)       | 2.2B   | ✗  | 6.84    | 2.63      | 1.18       | 1.73         | 1.31     | 25.78    |
| [HuBERT-XL](https://huggingface.co/nii-yamagishilab/hubert-xlarge-anti-deepfake) | 964M   | ✓  | 18.90   | 5.67      | 2.49       | 3.17         | 5.23     | 34.08    |
| [W2V-Small](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake) | 95M    | ✓  | 13.02   | 9.80      | 21.94      | 17.85        | 4.24     | 33.33    |
| [W2V-Large](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake) | 317M   | ✓  | 13.25   | 4.53      | 0.63       | 0.97         | 1.91     | 33.38    |
| [MMS-300M](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake)       | 317M   | ✓  | 7.93    | 2.27      | 1.35       | 5.92         | 2.90     | 32.80    |
| [MMS-1B](https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake)           | 965M   | ✓  | 9.06    | 2.56      | 1.22       | 1.73         | 1.82     | 27.70    |
| [XLS-R-1B](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake)       | 965M   | ✓  | 5.39    | 2.52      | 5.74       | 12.14        | 1.35     | 26.76    |
| [XLS-R-2B](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)       | 2.2B   | ✓  | 4.67    | 2.30      | 2.62       | 1.65         | 1.23     | 27.77    |


## Table of Contents
- [Try it out](#try-it-out)
- [Installation](#installation)
- [Usage demonstration](#usage-demonstration)
- [Usage in details](#usage)
- [Attribution and Licenses](#attribution-and-licenses)
- [Acknowledgments](#acknowledgments)

## Try it out

Full inference script is available on each model’s [Hugging Face](https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc) page. Simply copy some audio files and run the script to get their detection scores. 

## Installation
This setup is recommended if you plan to run custom experiments with the code.

```shell
### New conda environments ###
conda create --name antideepfake python==3.9.0
conda activate antideepfake
conda install pip==24.0

### Install PyTorch ###
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

### Install Fariseq ###
# pip install fairseq
# or to reproduce our venv:
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

### Clone Our Repository ###
# Please ensure that your AntiDeepfake/working directory 
# does not contain copies of fairseq or speechbrain repo.
```

## Usage demonstration

Here is a demonstration of using an AntiDeepfake checkpoint for deepfake detection.

```bash
# go to an empty project folder 

# create a Data folder and download a toy dataset
mkdir Data
cd Data
wget -O toy_example.tar.gz https://zenodo.org/records/7497769/files/project-04-toy_example.tar.gz
tar -xzvf toy_example.tar.gz
cd -

# git clone the code
git clone https://github.com/nii-yamagishilab/AntiDeepfake.git

# install dependency
bash AntiDeepfake/install.sh

# download an AntiDeepfake checkpoint
cd AntiDeepfake
mkdir downloads
wget -O downloads/mms_300m.ckpt https://zenodo.org/records/15580543/files/mms_300m.ckpt

# use venv created by install.sh
conda activate antideepfake

# scoring
python main.py inference hparams/mms_300m.yaml --base_path $PWD/.. --exp_name eval_antideepfake_mms_300m_toy_example --test_csv protocols/toy_example_test.csv --pretrained_weights '{"detector": "downloads/mms_300m.ckpt"}'

# ...
# INFO | __main__ | Loading pre-trained weights detector from downloads/mms_300m.ckpt
# 100%|███████████████████████████████████████████████████████████████████| 150/150 [00:03<00:00, 38.71it/s]
# Scores saved to ...

# computing metrics
python evaluation.py ../Log/exps/exp_mms_300m_eval_antideepfake_mms_300m_toy_example/evaluation_score.csv

# ...
# ===== METRICS SUMMARY =====
# For accuracy, precision, recall, f1, fpr and fnr, threshold of real class probablity is 0.5
#
#        roc_auc  accuracy  precision  recall      f1     fpr     fnr     eer  eer_threshold
#subset                                                                                     
#all         0.9935    0.9467     0.6818  0.9375  0.7895  0.0522  0.0625  0.0597   0.259
#ASV19LAdemo 0.9935    0.9467     0.6818  0.9375  0.7895  0.0522  0.0625  0.0597   0.259
```

For details on inference, post-training, and fine-tuning, please check the following section.

## Usage in details

### 0. Working directory structure

We assume the following project structure. This is provided as a reference to help understand how the code and data are organized:

```
/base_path/                 # The root of the project that contains data,
│                           # code, and experiment
│
├── Data/                   # Contains multiple databases
│   ├── ASVspoof2019-LA/    # Example database
│   ├── ASVspoof2021-DF/    # Example database
│   ├── ...                 # Other databases
│
├── fairseq/                # Directory for fairseq installation
├── speechbrain/            # Directory for speechbrain installation
│
├── Log/                    # Stores log files and model checkpoints
│   ├── exps/               # Directory for experiment outputs
│   ├── ssl-weights/        # Contains downloaded checkpoint files
│
│
├── AntiDeepfake/           # This AntiDeepfake repository
│   ├── hparams             # Configuration files
│   │   ├── xx.yaml         # 
│   ├── ...                 # Code files
│   ├── protocols/
│   │   ├── xx.py           # python script for generating protocols
│   │   ├── train.csv       # The generated training set protocol
│   │   ├── valid.csv       # The generated validation set protocol
│   │   ├── test.csv        # The generated test set protocol

```

The folder structure can be altered. By doing so, please remember to change the path variables in `hparams/*.yaml`.

### 1. Generate protocols

Training and inference scripts provided in this repository are designed to load audio files listed in train/valid/test CSV protocol files.

In the demonstration above, we used `protocols/toy_example_test.csv` to do inference. You can check the content of this CSV file.

Python scripts for generating database protocols are provided in `protocols`. Each script is named after the database it processes. 

All protocols are designed to follow the same format so we can easily shuffle, split, or merge them. To merge multiple CSV protocols as in our experiment, you can refer to `generate_protocol_by_proportion.py`.

To generate protocols for your own data:

- refer to `toy_example.py` and the downloaded toy_example.tar.gz for example.
- refer to `ASVspoof2019-LA.py` if you have a protocol file with ground truth labels for each audio file.
- refer to `CVoiceFake.py` if your real and fake audios are stored separately.
- refer to `WildSVDD.py` if your audio filenames indicate whether they are real or fake.

### 2. Download checkpoints

Our code creates the models using configuration files. Their front ends are initialized with random weights.

* If you do inference without post-training or fine-tuning, please download an AntiDeepfake checkpoint (see example in Usage Demonstration above). The weights of the front end and the rest of the model will be overwritten using the AntiDeepfake checkpoint.

* If you do post-training or fine-tuning upon an AntiDeepfake checkpoint, please download the AntiDeepfake checkpoint. The weights of the front end and the rest of the model will be overwritten using the AntiDeepfake checkpoint.

* If you want to do your own post-training using Fairseq pre-trained SSL front ends, please download Fairseq checkpoints. The weights of the front end will be re-initialized using the Fairseq checkpoint.


| Model           | Download Link                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AntiDeepfake**      | [Zenodo](https://zenodo.org/records/15580543) and [Hugging Face](https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc) |
| **Fairseq MMS**         | Pretrained models from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#pretrained-models)                                                                      |
| **Fairseq XLS-R**       | Model link from [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md#xls-r)                                                          |
| **Fairseq Wav2Vec 2.0** | Base (no finetuning) and Large (LV-60 + CV + SWBD + FSH), no finetuning, from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#pre-trained-models)                |
| **Fairseq HuBERT**      | Extra Large (\~1B params), trained on Libri-Light 60k hrs, no finetuning, from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#pre-trained-and-fine-tuned-asr-models) |


### 3. Training

To post-train or fine-tune upon an AntiDeepfake checkpoint (e.g., MMS-300M):
```
python main.py hparams/mms_300m.yaml \
    --base_path /your/base_path \
    --exp_name fine_tuning \
    --lr 1e-6 \
    # Perform full validation every 2025 mini-batches
    --valid_step 2025 \
    # Enable RawBoost data augmentation
    --use_da True \
    # Initialize model weights with AntiDeepfake checkpoint
    --pretrained_weights '{"detector": "/path/to/your/downloaded/antideepfake/mms_300m.ckpt"}'
```

To start post-train with a Fairseq checkpoint (e.g., MMS-300M):
```
python main.py hparams/mms_300m.yaml \
    --base_path /your/base_path \
    --exp_name post_training \
    --lr 1e-7 \
    --valid_step 100000 \
    # Disable RawBoost data augmentation
    --use_da False \
    # Initialize model weights with Fairseq checkpoint (default setting)
    --pretrained_weights '{"detector": "/base_path/Log/ssl-weights/base_300m.pt"}'
```

Notes:
1. Configuration YAML files are named after the model they correspond to. Please use the corresponding configuration file in `hparams`.
2. Training logs and checkpoints will be saved under `/base_path/Log/exps/exp_mms_300m_<exp_name>`.
3. If the above `exp` folder already exists, the script will try to resume training from the last saved checkpoint in the folder.
4. For multi-GPU training, please use:
```bash
torchrun --nnodes=1 --nproc-per-node=<NUM_GPU> main.py hparams/<MODEL>.yaml
```

### 4. Inference and evaluation

#### Inference (generating CSV score)

For using the best validation checkpoint from your own experiment:
```
python main.py inference hparams/mms_300m.yaml \
    --base_path /your/base_path \
    # Exp folder name must match the name used during training
    --exp_name fine_training \
    --test_csv /path/to/your/test.csv
```
The script will automatically search for the best validation checkpoint in the specified experiment folder `/base_path/Log/exps/exp_mms_300m_<exp_name>`. It will generate an `evaluation_score.csv` file in the same folder.


For using AntiDeepfake checkpoints without training:
```
python main.py inference hparams/mms_300m.yaml \
    --base_path /your/base_path \
    # Use a new exp folder name to avoid conflicts
    --exp_name eval_antideepfake_mms_300m \
    --test_csv /path/to/your/test.csv
    # Initialize model weights with AntiDeepfake checkpoint
    --pretrained_weights '{"detector": "/path/to/your/downloaded/antideepfake/mms_300m.ckpt"}'
```

#### Evaluating CSV score
Run:
```
python evaluation.py /path/to/your/evaluation_score.csv
```
You will get results similar to this:
```
===== METRICS SUMMARY =====
For accuracy, precision, recall, f1, fpr and fnr, threshold of real class probablity is 0.5

        roc_auc  accuracy  precision  recall      f1     fpr     fnr     eer  eer_threshold
subset                                                                                     
all         0.9935    0.9467     0.6818  0.9375  0.7895  0.0522  0.0625  0.0597          0.259
ASV19LAdemo 0.9935    0.9467     0.6818  0.9375  0.7895  0.0522  0.0625  0.0597          0.259
...
```

The row `all` show results computed over all the scores in the file. 
The rows below `all` list the results for each subset in the file, where the subset is identified by the file ID prefix (e.g., ASV19LAdemo in Usage demonstration).

# Our fine-tuning results

The released AntiDeepfake checkpoints are post-trained checkpoints. They can be fine-tuned to a specific task in a specific domain.

Described below is the performance of fine-tuning AntiDeepfake models on Deepfake-Eval-2024 train set (PT = Pre-training, PST = Post-training, FT = Fine-tuning, 4s = Input Duration is 4 seconds).

Please note that we do not provide these fine-tuned checkpoints.

| Model ID     | PT+PST+FT 4s | PT+PST+FT 10s | PT+PST+FT 13s | PT+PST+FT 30s | PT+PST+FT 50s | PT+FT 4s | PT+FT 10s | PT+FT 13s | PT+FT 30s | PT+FT 50s |
|--------------|--------------|---------------|---------------|---------------|---------------|----------|-----------|-----------|-----------|-----------|
| W2V-Large    | 19.56        | 12.10         | 10.94         | 10.52         | 11.37         | 24.42    | 22.46     | 22.14     | 21.15     | 21.51     |
| MMS-300M     | 17.15        | 13.37         | 12.31         | 11.05         | 10.75         | 19.77    | 13.29     | 12.77     | 12.01     | 12.29     |
| MMS-1B       | 12.11        | 10.36         | 10.03         | 8.61          | 9.37          | 19.86    | 10.32     | 11.55     | 11.05     | 11.52     |
| XLS-R-1B     | 11.85        | 10.00         | 9.27          | 8.50          | 8.29          | 19.95    | 17.18     | 16.31     | 10.63     | 11.21     |
| XLS-R-2B     | 12.14        | 9.80          | 9.98          | 9.46          | 9.68          | 12.88    | 10.75     | 10.39     | 9.67      | 9.98      |

# **Attribution and Licenses**
All AntiDeepfake models were developed by [Yamagishi Lab](https://yamagishilab.jp/) at the National Institute of Informatics (NII), Japan. All model weights and code scripts are intellectual property of NII and are made available for research and educational purposes under the licenses
* **Code** – BSD-3-Clause, see [`LICENSE-CODE`](./LICENSE-CODE).
* **Model checkpoints** – CC BY-NC-SA 4.0, see [`LICENSE-CHECKPOINT`](./LICENSE-CHECKPOINT).

# **Acknowledgments**
This project is based on results obtained from project JPNP22007, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

It is also partially supported by the following grants from the Japan Science and Technology Agency (JST):
- AIP Acceleration Research (Grant No. JPMJCR24U3)
- PRESTO (Grant No. JPMJPR23P9)

This study was carried out using the TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

Codes are based on the implementations of [wav2vec 2.0 pretraining with SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/self-supervised-learning/wav2vec2) and [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts).
## **Citation**
If you find this repository useful, please consider citing:
```
@article{antideepfake_2025,
      title={Post-training for Deepfake Speech Detection}, 
      author={Wanying Ge, Xin Wang, Xuechen Liu, Junichi Yamagishi},
      journal={arXiv preprint arXiv:2506.21090},
      year={2025},
}
```
