# Training with Weights & Biases

`main.py` supports optional W&B tracking; the YAML files are unchanged.
Tracking helpers live in `utils_wandb.py`.
W&B is disabled by default: explicitly pass `--wandb-mode online` or
`--wandb-mode offline` to enable it. Install W&B in the existing training environment:

```bash
pip install wandb
wandb login
python main.py hparams/w2v_small.yaml \
  --base_path /path/to/experiment \
  --wandb-mode online \
  --wandb-project AntiDeepfake \
  --wandb-name w2v-small-try
```

Use your existing dataset, pretrained-weight, and other YAML overrides as usual.
Optional flags: `--wandb-entity TEAM`, `--wandb-project PROJECT`,
`--wandb-name NAME`, and `--wandb-mode online|offline|disabled`.
Project, entity, and name defaults honor `WANDB_PROJECT`, `WANDB_ENTITY`, and
`WANDB_NAME`. `WANDB_MODE` does not enable tracking; the CLI mode defaults to
`disabled` regardless of the environment. Do not put API keys in your YAML or source code; use `wandb login`
or the `WANDB_API_KEY` environment variable.

Saved information:

- Hyperparameters: resolved scalar/list/dictionary values in W&B config and
  `<output_folder>/wandb_hparams.json`. Instantiated objects are represented by
  class names; partial callables include arguments. The original YAML and
  SpeechBrain's saved `hyperparams.yaml` retain construction details.
- Metrics: training losses and learning rate at the configured `log_interval`,
  plus validation loss and EER at each validation. Charts use optimizer `steps`.
- Checkpoints: SpeechBrain retains its best checkpoint by validation EER locally.
  Each newly selected best checkpoint is uploaded as a version of the W&B model
  artifact `checkpoint-RUN_ID`, with `best` and `latest` aliases. The whole
  checkpoint directory is included, preserving the training state saved by
  SpeechBrain. W&B retains earlier artifact versions even when local checkpoints
  are pruned; large models can use substantial upload bandwidth and storage.
- Logs: console output from the tracked execution is appended to
  `<output_folder>/output.log`. This file, existing text training logs, configuration
  files, `main.py`, and `utils_wandb.py` are uploaded when execution ends, including on ordinary
  Python exceptions. A forced kill cannot execute final uploads. Existing
  TensorBoard logging continues locally.

Only the main distributed process creates a W&B run. Every invocation creates a
new run; SpeechBrain's existing local checkpoint recovery still handles training
resumption. To restore a downloaded artifact, place its complete contents in a
`CKPT+...` subdirectory of the YAML-configured `save_folder` before starting.

For a cluster without internet access:

```bash
python main.py hparams/w2v_small.yaml \
  --base_path /path/to/experiment --wandb-mode offline
# Later, from a machine with access to the saved run and internet:
wandb sync /path/to/output_folder/wandb/offline-run-*
```

`--wandb-mode disabled` (the default) requires no W&B installation and skips
all tracking helpers, including the extra console log and configuration snapshot.
The original SpeechBrain logging and checkpoint behavior remains active. Existing `inference` and `analysis`
commands are supported; checkpoint artifact uploads occur during training.

API reference: https://docs.wandb.ai/ref/python/experiments/run/
