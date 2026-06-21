<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/nii-yamagishilab/AntiDeepfake/blob/main/logo.png?raw=true" width="60%" alt="AntiDeepfake" />
</div>

## Introduction to GRPO

We explore reinforcement-learning-style fine-tuning for deepfake detection. Instead of the standard supervised cross-entropy (CE), we adapt **Group Relative Policy Optimization (GRPO)** to the binary real/fake classifier.

This is detailed in paper
```
Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?
Xin Wang, Wanying Ge, Junichi Yamagishi
in INTERSPEECH 2026
```

## Implementation

What we need to run a GRPO training step:
1. A pre-trained AntiDeepfake detector to fine-tune
2. A frozen copy of that detector as the reference model
3. (optional) An old-policy copy for the clipped ratio in PPO-style updates
4. The GRPO loss functions

### GRPO loss functions

The GRPO objectives are implemented in [`algo/rl.py`](algo/rl.py):

| Function | `rl_config.algo` | Description |
| --- | --- | --- |
| `grpo2_loss`         | `grpo2`          | GRPO with a reward-weighted policy-gradient term and a KL penalty against the reference model. |
| `grpo3_loss`         | `grpo3`          | PPO-style GRPO that additionally clips the probability ratio between the current and *old* policy (needs `old_detector`). |
| `grpo3_degraded_ref` | `grpo3_degraded` | Ablation variant that degrades GRPO into an SFT-like objective (no reward normalization / no ratio). |

Each loss takes the current logits, the reference logits, the sampled labels
(`sa_labels`), the ground-truth labels, and the `rl_config` dictionary. Sampling is
done in [`models/W2V.py`](models/W2V.py) via `forward_grpo`, which draws
`sample_num` samples per utterance from the softmax over the detector logits.
The reward is a simple indicator (`1` if a sample matches the ground truth, `0`
otherwise), normalized within the group before weighting.

### Configuration files

Experiment configs live under [`hparams/ft_grpo/`](hparams/ft_grpo/). They are
organized per SSL front-end (`w2v_small`, `w2v_large`, `mms_300m`, `mms_1b`,
`xlsr_1b`, `xlsr_2b`) and per training paradigm:

| Suffix | Paradigm |
| --- | --- |
| `<model>_pre-post.yaml`         | Pre/post-training baseline (`rl_config.algo: null`, plain CE). |
| `<model>_pre-post-sft_{1,2,3}.yaml`     | Supervised (CE) fine-tuning baseline, three repeated runs. |
| `<model>_pre-post-grpo_v1_{1,2,3}.yaml` | GRPO with the `grpo3` (clipped, old-policy) objective, three repeated runs. |
| `<model>_pre-post-grpo_v2_{1,2,3}.yaml` | GRPO with the `grpo2` objective, three repeated runs. |

The `_1` / `_2` / `_3` suffixes are repeated runs used to average results.

The key extra block in a GRPO config is `rl_config`, for example:
```yaml
rl_config:
   algo: grpo3        # grpo2 / grpo3 / grpo3_degraded / null
   sample_num: 64     # samples drawn per utterance
   std_floor: 1e-5    # floor added to reward std when normalizing
   beta: 0.04         # KL penalty weight
   inner_step_size: 1000  # (grpo3 only) steps between old-policy refresh
   epsilon: 0.2           # (grpo3 only) PPO clipping range
```
A GRPO config also instantiates a `ref_detector` (and, for `grpo3`, an
`old_detector`) mirroring the main `ssl` model.

* Experiment setup (see [`hparams/ft_grpo/README.txt`](hparams/ft_grpo/README.txt)):
    * Training: segmented DF-2024 train set
    * Validation: asvspoof2019-la-dev

### Running training

Use the same entry point as the rest of AntiDeepfake (see [README.md](README.md)),
just pointing at a `ft_grpo` config:
```bash
# GRPO (v2 / grpo2) fine-tuning of the W2V-small detector
python main.py hparams/ft_grpo/w2v_small_pre-post-grpo_v2_1.yaml \
    --base_path /your/base_path \
    --exp_name grpo_v2_run1 \
    --ckpt_path /path/to/your/antideepfake/w2v_small.ckpt

# SFT (CE) baseline for comparison
python main.py hparams/ft_grpo/w2v_small_pre-post-sft_1.yaml \
    --base_path /your/base_path \
    --exp_name sft_run1 \
    --ckpt_path /path/to/your/antideepfake/w2v_small.ckpt
```

Notes:
1. Training logs and checkpoints are saved under
   `/base_path/Log/exps/exp_<model>_<exp_name>`.
2. If that folder already exists, the script resumes from the last checkpoint.
3. The reference (and old) policy are copied from the detector weights at
   initialization, so make sure `--ckpt_path` points at the checkpoint you want to
   fine-tune from.
