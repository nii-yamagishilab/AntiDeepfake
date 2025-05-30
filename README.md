<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/logo.png?raw=true" width="60%" alt="AntiDeepfake" />
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/nii-yamagishilab"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NII Yamagishi Lab-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/LICENSE-CODE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-BSD 3-f5de53?&color=f5de53"/></a>
  <a href="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/LICENSE-CHECKPOINT"><img alt="Model License"
    src="https://img.shields.io/badge/Checkpoint_License-CC BY%20NC%20SA 4.0-f5de53?&color=f5de53"/></a>
<a href="https://zenodo.org/record/your_record_id">
  <img alt="Zenodo" src="https://img.shields.io/badge/%20Zenodo-Pretrained%20Checkpoints-0077C8?logo=zenodo&logoColor=white" />
</a>

  <br>
  <a href="https://arxiv.org"><b>Paper Link</b></a>
</div>

<hr>

## Introduction

The AntiDeepfake project provides a series of powerful foundation models post-trained for deepfake detection. The AntiDeepfake model can be used for feature extraction for deepfake detection in a zero-shot manner, or it may be further fine-tuned and optimized for a specific database or deepfake-related task.

This table below summarizes performance across multiple evaluation datasets, along with their sizes, to help guide your selection.

For more technical details and analysis, please refer to our paper [Post-training for Deepfake Speech Detectio](paper-link).

| 🤗 Model                                                                                 | Params | RawBoost | ADD2023 | DEEP-VOICE | FakeOrReal | FakeOrReal-Norm | In-the-Wild | Deepfake-Eval-2024 |
|------------------------------------------------------------------------------------------|--------|----|---------|-----------|------------|--------------|----------|
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
| [MMS-1B](https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake)           | 965M   | ✓  | 9.06    | 2.56      | 1.22       | 1.73         | 1.82     | 23.70    |
| [XLS-R-1B](https://huggingface.co/nii-yamagishilab/xls-r-1b-anti-deepfake)       | 965M   | ✓  | 5.39    | 2.52      | 5.74       | 12.14        | 1.35     | 26.76    |
| [XLS-R-2B](https://huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake)       | 2.2B   | ✓  | 4.67    | 2.30      | 2.62       | 1.65         | 1.23     | 27.77    |


## 📢 News and Updates


## Table of Contents
- [Try it out](#try-it-out)
- [Installation](#installation)
- [Usage](#usage)
- [Attribution and Licenses](#attribution-and-licenses)
- [Acknowledgments](#acknowledgments)

## Try it out

Inference code is available on each model’s [Hugging Face](https://huggingface.co/nii-yamagishilab) page. 

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
# fairseq 0.10.2 on pip does not work
git clone https://github.com/pytorch/fairseq
cd fairseq
# checkout this specific commit. Latest commit does not work
git checkout 862efab86f649c04ea31545ce28d13c59560113d
pip install --editable .
cd ../

### Install SpeechBrain ###
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .

### Install other packages ###
pip install tensorboard tensorboardX soundfile pandarallel scikit-learn numpy==1.21.2 pandas==1.4.3 scipy==1.7.2
```

Additionally, to train or run `W2V_Small`, `W2V_Large` and `HuBERT_XL`, you need to update line 315 in `fairseq/fairseq/checkpoint_utils.py` to:
```
state = torch.load(f, map_location=torch.device("cpu"), weights_only=False) 
```


## Usage

### 0. Working directory structure
Below is an overview of our working directory structure. This is provided as a reference to help understand how the code and data are organized:

```
/base_path/
│
├── Data/                   # Contains multiple databases
│   ├── ASVspoof2019-LA/    # Example database
│   ├── ASVspoof2021-DF/    # Example database
│   ├── ...                 # Other databases
│
├── fairseq/                # Directory for fairseq installation
│
├── Log/                    # Stores log files and model checkpoints
│   ├── exps/               # Directory for experiment outputs
│   ├── ssl-weights/        # Contains downloaded fairseq SSL checkpoint files (ssl.pt)
│
├── speechbrain/            # Directory for speechbrain installation
│
├── AntiDeepfake/       # This repository
│   ├── ...                 # Project files
│   ├── protocols/
│   │   ├── xx.py           # python script for generating protocols
│   │   ├── train.csv       # The generated training set protocol
│   │   ├── valid.csv       # The generated validation set protocol
│   │   ├── test.csv        # The generated test set protocol

```
### 1. Download pretrained Fariseq checkpoints

Please download the pretrained SSL checkpoints from Fairseq GitHub repo to your `/base_path/Log/ssl-weights/`. These checkpoints are used to build the front-end SSL architecture in this project.

Please note that this step is not for downloading our AntiDeepfake checkpoints.

| Model           | Download Link                                                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MMS**         | Pretrained models from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#pretrained-models)                                                                      |
| **XLS-R**       | Model link from [here](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md#xls-r)                                                          |
| **Wav2Vec 2.0** | Base (no finetuning) and Large (LV-60 + CV + SWBD + FSH), no finetuning, from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#pre-trained-models)                |
| **HuBERT**      | Extra Large (\~1B params), trained on Libri-Light 60k hrs, no finetuning, from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#pre-trained-models) |


### 2. Generate protocols

Protocol generation scripts are provided in `protocols/`. Each script is named after the database it processes.

For large databases such as SpoofCeleb and MLS, protocol generation may take several hours to days.

To merge multiple protocols, you can refer to `generate_protocol_by_proportion.py`.

To generate protocols for your own data:

- Refer to `ASVspoof2019-LA.py` if you have a protocol file with ground truth labels for each audio file.
- Refer to `CVoiceFake.py` if your real and fake audios are stored separately.
- Refer to `WildSVDD.py` if your audio filenames indicate whether they are real or fake.

### 3. Train

Example bash command to train the MMS-300M model:

```
python main.py hparams/mms_300m.yaml \
    --exp_name my_job \
    --lr 1e-6 \
    --use_da True      # Enable RawBoost data augmentation
```
Training logs and checkpoints will be saved under `/base_path/Log/exps/exp_mms_300m_my_job`.  
Evaluation results with the best validation model will be stored in the same folder with the name `evaluation_score.csv`.

You can use the following script for multi-GPU training
```
torchrun --nnodes=1 --nproc-per-node=NUM_GPU main.py hparams/<MODEL>.yaml
```
### 4. Performance evaluation

#### With CSV score
```
python evaluation.py /base_path/Log/exps/exp_mms_300m_my_job/evaluation_score.csv
```
You will get results similar to this:
```
No data for ID_PREFIX_1
No data for ID_PREFIX_2

===== METRICS SUMMARY =====
        roc_auc  accuracy  precision  recall      f1     fpr     fnr     eer  eer_threshold
subset                                                                                     
all       0.951    0.8879     0.9421  0.8823  0.9112  0.1016  0.1177  0.1079         0.4114
```
Results are shown for each subset and also the full set listed in your protocol.

The message "No data for ID_PREFIX\_1" means no entry IDs in your protocol start with `ID_PREFIX_1`. Each ID should begin with a dataset-specific `ID_PREFIX`, set during its protocol generation.

#### Evaluate from a checkpoint
You can use `test.py` for standalone score generation with any model checkpoint, or to evaluate on data not included in your test protocol. Please refer to its docstring for detailed usage instructions.


### 5. Further fine-tuning

We provide our .ckpt checkpoints on [Zenodo](https://zenodo.org/), to continue fine-tuning with these checkpoints, please use the same training script from Step 3 and:
  - Set `use_pretrained=True`
  - Set `pretrained_weights.detector` to the path of the .ckpt file

Fine-tuning will follow a similar process to training a new model, except that SSL weights will be initialized as AntiDeepfake checkpoints.

Below is our evaluation performance of fine-tuning AntiDeepfake models on Deepfake-Eval-2024 train set (PT = Pre-training, PST = Post-training, FT = Fine-tuning).

| Model ID     | PT+PST+FT 4s | PT+PST+FT 10s | PT+PST+FT 13s | PT+PST+FT 30s | PT+PST+FT 50s | PT+FT 4s | PT+FT 10s | PT+FT 13s | PT+FT 30s | PT+FT 50s |
|--------------|--------------|---------------|---------------|---------------|---------------|----------|-----------|-----------|-----------|-----------|
| W2V-Large    | 19.56        | 12.10         | 10.94         | 10.52         | 11.37         | 24.42    | 22.46     | 22.14     | 21.15     | 21.51     |
| MMS-300M     | 17.15        | 13.37         | 12.31         | 11.05         | 10.75         | 19.77    | 13.29     | 12.77     | 12.01     | 12.29     |
| MMS-1B       | 12.11        | 10.36         | 10.03         | 8.61          | 9.37          | 19.86    | 10.32     | 11.55     | 11.05     | 11.52     |
| XLS-R-1B     | 11.85        | 10.00         | 9.27          | 8.50          | 8.29          | 19.95    | 17.18     | 16.31     | 10.63     | 11.21     |
| XLS-R-2B     | 12.14        | 9.80          | 9.98          | 9.46          | 9.68          | 12.88    | 10.75     | 10.39     | 9.67      | 9.98      |

# **Attribution and Licenses**
All AntiDeepfake models were developed by [Yamagishi Lab](https://yamagishilab.jp/) at the National Institute of Informatics (NII), Japan. All model weights are the intellectual property of NII and are made available for research and educational purposes under the licenses
* **Code** – BSD-3-Clause, see [`LICENSE-CODE`](./LICENSE-CODE).
* **Model checkpoints** – CC BY-NC-SA 4.0, see [`LICENSE-CHECKPOINT`](./LICENSE-CHECKPOINT).

# **Acknowledgments**
This project is based on results obtained from project JPNP22007, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).

It is also partially supported by the following grants from the Japan Science and Technology Agency (JST):
- AIP Acceleration Research (Grant No. JPMJCR24U3)
- PRESTO (Grant No. JPMJPR23P9)

This study was carried out using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.

Codes are based on the implementations of [wav2vec 2.0 pretraining with SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/self-supervised-learning/wav2vec2) and [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts).
## **Citation**
```
@misc{paper,
      title={TITLE}, 
      author={Wanying Ge, Xin Wang, Xuechen Liu, Junichi Yamagishi},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={},
      url={}, 
}
```
