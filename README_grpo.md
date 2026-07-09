<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

## Introduction to GRPO Fine-tuning

We investigate whether fine-tuning by reinforcement learning improves generalization in speech deepfake detection. We fine-tune the post-trained models using Group Relative Policy Optimization (GRPO).

This is detailed in paper
```
Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?
Xin Wang, Wanying Ge, Junichi Yamagishi
INTERSPEECH 2026
```

## Implementation

What we need to run GRPO fine-tuning:
1. A post-trained AntiDeepfake checkpoint (see [README.md](./README.md) for download links), which initializes the policy model, the frozen reference model, and the old policy model
2. Train/validation/test CSV protocols (see [README.md](./README.md) on how to generate them)
3. A configuration YAML file from `hparams/ft_grpo`

The GRPO losses are implemented in `algo/rl.py`. For each input utterance, the model samples `sample_num` labels from its predicted real/fake probabilities (see `forward_grpo` in `models/W2V.py`), a binary reward of 1 or 0 is assigned to each sampled label by comparing it with the ground truth, and the group-normalized rewards are used to optimize the policy, with a KL-divergence penalty against the frozen reference model.

### Configurations

Configuration files in `hparams/ft_grpo` are named `<model>_<method>.yaml`, where `<model>` is one of the post-trained models (`mms_300m`, `mms_1b`, `w2v_small`, `w2v_large`, `xlsr_1b`, `xlsr_2b`):

| Configuration | `rl_config: algo` | Method in the paper | Description |
|---------------|-------------------|---------------------|-------------|
| `<model>_pre-post-grpo_v1.yaml` | `grpo3` | GRPO | GRPO with clipped policy ratio (`epsilon: 0.2`) between the current and the old policy model. The old policy model is refreshed every `inner_step_size` optimizer steps. Loss function: `grpo3_loss` |
| `<model>_pre-post-grpo_v2.yaml` | `grpo2` | GRPO<sub>s</sub> | Simplified GRPO without the clipped policy ratio and the old policy model, which updates the model after every batch of sampled data. Loss function: `grpo2_loss` |
| `<model>_pre-post-sft.yaml` | (none) | SFT | SFT baseline using the standard cross-entropy loss |
| `<model>_pre-post.yaml` | `null` | - | For inference using the post-trained checkpoint without any fine-tuning |

In the paper, each configuration was fine-tuned three times independently. The runs share the same configuration file -- please use a different `<outputfolder>` for each run.

Key hyper-parameters in `rl_config`:
* `sample_num: 64`: number of labels sampled per utterance (i.e., the group size)
* `beta: 0.04`: weight of the KL-divergence penalty against the frozen reference model
* `epsilon: 0.2`: clipping range of the policy ratio (`grpo3` only)
* `inner_step_size: 1000`: number of optimizer steps between two updates of the old policy model (`grpo3` only)

In the paper experiments, we used the segmented Deepfake-Eval-2024 train set for fine-tuning and the ASVspoof 2019 LA dev set for validation.

### Training

To fine-tune a post-trained AntiDeepfake checkpoint (e.g., MMS-300M) with GRPO:
```bash
python main.py hparams/ft_grpo/mms_300m_pre-post-grpo_v1.yaml \
    --base_path <basepath> \
    --data_folder <datafolder> \
    --output_folder <outputfolder> \
    --pretrained_weights '{"detector": "<pretrained>"}' \
    --train_csv <train_protocol> \
    --valid_csv <valid_protocol>
```

To run the GRPO variant without the clipped policy ratio, or the SFT baseline, simply switch the configuration file to `mms_300m_pre-post-grpo_v2.yaml` or `mms_300m_pre-post-sft.yaml`.

### Inference and evaluation

After fine-tuning, run inference on a test set using the same configuration file and `<outputfolder>`:
```bash
python main.py inference hparams/ft_grpo/mms_300m_pre-post-grpo_v1.yaml \
    --base_path <basepath> \
    --data_folder <datafolder> \
    --output_folder <outputfolder> \
    --pretrained_weights '{"detector": "<pretrained>"}' \
    --test_csv <test_protocol> \
    --score_path <outputfolder>/<output_file_name>.csv
```
The script will automatically search for the best validation checkpoint in `<outputfolder>` and save the scores to `<outputfolder>/<output_file_name>.csv`.

To evaluate the post-trained checkpoint without fine-tuning as a reference, use the `<model>_pre-post.yaml` configuration and a new `<outputfolder>`.

Finally, compute the metrics from the saved score file:
```bash
python evaluation.py <outputfolder>/<output_file_name>.csv
```
