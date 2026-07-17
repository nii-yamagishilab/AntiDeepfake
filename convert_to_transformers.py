"""Convert AntiDeepfake checkpoints from fairseq to native HuggingFace transformers format.

This script converts .ckpt files (which contain fairseq Wav2Vec2 weights) to a format
loadable by transformers.Wav2Vec2Model, eliminating the fairseq dependency for inference.

Usage:
    python convert_to_transformers.py --model mms_300m --ckpt path/to/mms_300m.ckpt --output converted/mms_300m/
    python convert_to_transformers.py --model mms_300m --safetensors path/to/model.safetensors --output converted/mms_300m/

Supported models: w2v_small, w2v_large, mms_300m, mms_1b, xlsr_1b, xlsr_2b

After conversion, use with:
    from models.W2V_transformers import Model
    model = Model('mms_300m')
    state = torch.load('converted/mms_300m/full_model.pt')
    model.load_state_dict(state)
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config

HF_MODEL_IDS = {
    'w2v_small': 'facebook/wav2vec2-base',
    'w2v_large': 'facebook/wav2vec2-large',
    'mms_300m': 'facebook/mms-300m',
    'mms_1b': 'facebook/mms-1b-all',
    'xlsr_1b': 'facebook/wav2vec2-xls-r-1b',
    'xlsr_2b': 'facebook/wav2vec2-xls-r-2b',
}

MODEL_DIMS = {
    'w2v_small': 768, 'w2v_large': 1024, 'mms_300m': 1024,
    'mms_1b': 1280, 'xlsr_1b': 1280, 'xlsr_2b': 1920,
}

# Per-model feature-extractor config overrides.
#
# The NII AntiDeepfake fairseq checkpoints are all trained with
# feat_extract_norm="layer" + conv_bias=True (LayerNorm + bias on every conv
# in the feature extractor). For two HF base models - wav2vec2-base,
# wav2vec2-large - the default HF config uses feat_extract_norm="group" +
# conv_bias=False, so building Wav2Vec2Model(config) from the HF default
# yields a feature extractor that doesn't match the converted state_dict.
# Override the config explicitly for those two models so the built
# architecture matches the checkpoint.
#
# mms_1b (facebook/mms-1b-all) is NOT listed here: its HF base config is
# already feat_extract_norm="layer" + conv_bias=True. The separate
# load_state_dict failure reported on #14 for mms_1b is caused by the
# adapter layers present in mms-1b-all (it is an ASR fine-tune, not the
# pure SSL backbone), and needs a different fix than this feature-extractor
# override. Tracking that separately.
OVERRIDE_FE_CONFIG = {
    'w2v_small': {'feat_extract_norm': 'layer', 'conv_bias': True},
    'w2v_large': {'feat_extract_norm': 'layer', 'conv_bias': True},
}


def build_config(model_name):
    """Load the HF base config, then apply per-model overrides."""
    config = Wav2Vec2Config.from_pretrained(HF_MODEL_IDS[model_name])
    for k, v in OVERRIDE_FE_CONFIG.get(model_name, {}).items():
        setattr(config, k, v)
    return config


MODEL_LAYERS = {
    'w2v_small': 12, 'w2v_large': 24, 'mms_300m': 24,
    'mms_1b': 48, 'xlsr_1b': 48, 'xlsr_2b': 48,
}

N_CONV_LAYERS = 7
PRETRAINING_PREFIXES = ('final_proj.', 'project_q.', 'quantizer.')


def build_key_map(model_name):
    """Build mapping from fairseq weight keys to transformers keys."""
    n_layers = MODEL_LAYERS[model_name]
    key_map = {}

    for i in range(N_CONV_LAYERS):
        fs = f"feature_extractor.conv_layers.{i}"
        tf = f"feature_extractor.conv_layers.{i}"
        key_map[f"{fs}.0.weight"] = f"{tf}.conv.weight"
        key_map[f"{fs}.0.bias"] = f"{tf}.conv.bias"
        key_map[f"{fs}.2.1.weight"] = f"{tf}.layer_norm.weight"
        key_map[f"{fs}.2.1.bias"] = f"{tf}.layer_norm.bias"

    key_map["layer_norm.weight"] = "feature_projection.layer_norm.weight"
    key_map["layer_norm.bias"] = "feature_projection.layer_norm.bias"
    key_map["post_extract_proj.weight"] = "feature_projection.projection.weight"
    key_map["post_extract_proj.bias"] = "feature_projection.projection.bias"

    key_map["encoder.pos_conv.0.weight_g"] = "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    key_map["encoder.pos_conv.0.weight_v"] = "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    key_map["encoder.pos_conv.0.bias"] = "encoder.pos_conv_embed.conv.bias"

    key_map["encoder.layer_norm.weight"] = "encoder.layer_norm.weight"
    key_map["encoder.layer_norm.bias"] = "encoder.layer_norm.bias"

    for i in range(n_layers):
        fs = f"encoder.layers.{i}"
        tf = f"encoder.layers.{i}"
        for proj in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            key_map[f"{fs}.self_attn.{proj}.weight"] = f"{tf}.attention.{proj}.weight"
            key_map[f"{fs}.self_attn.{proj}.bias"] = f"{tf}.attention.{proj}.bias"
        key_map[f"{fs}.self_attn_layer_norm.weight"] = f"{tf}.layer_norm.weight"
        key_map[f"{fs}.self_attn_layer_norm.bias"] = f"{tf}.layer_norm.bias"
        key_map[f"{fs}.final_layer_norm.weight"] = f"{tf}.final_layer_norm.weight"
        key_map[f"{fs}.final_layer_norm.bias"] = f"{tf}.final_layer_norm.bias"
        key_map[f"{fs}.fc1.weight"] = f"{tf}.feed_forward.intermediate_dense.weight"
        key_map[f"{fs}.fc1.bias"] = f"{tf}.feed_forward.intermediate_dense.bias"
        key_map[f"{fs}.fc2.weight"] = f"{tf}.feed_forward.output_dense.weight"
        key_map[f"{fs}.fc2.bias"] = f"{tf}.feed_forward.output_dense.bias"

    key_map["mask_emb"] = "masked_spec_embed"
    return key_map


def convert_checkpoint(model_name, src_state_dict, output_dir):
    """Convert fairseq state dict to transformers format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ssl_prefix = "m_ssl.model."
    ssl_keys = {}
    head_keys = {}

    for k, v in src_state_dict.items():
        if k.startswith(ssl_prefix):
            ssl_keys[k[len(ssl_prefix):]] = v
        elif k.startswith("proj_fc.") or k.startswith("adap_pool1d."):
            head_keys[k] = v

    print(f"  Source: {len(src_state_dict)} total, {len(ssl_keys)} SSL, {len(head_keys)} head")

    key_map = build_key_map(model_name)
    new_state = {}
    mapped = 0
    skipped = []
    unmapped = []

    for fs_key, tensor in ssl_keys.items():
        if fs_key in key_map:
            new_state[key_map[fs_key]] = tensor
            mapped += 1
        elif any(fs_key.startswith(p) for p in PRETRAINING_PREFIXES):
            skipped.append(fs_key)
        else:
            unmapped.append(fs_key)

    print(f"  Mapped: {mapped}/{len(ssl_keys)} SSL keys")
    if skipped:
        print(f"  Skipped (pretraining-only): {len(skipped)}")
    if unmapped:
        print(f"  Unmapped: {len(unmapped)}")
        for k in unmapped:
            print(f"    {k} [{ssl_keys[k].shape}]")

    hf_id = HF_MODEL_IDS[model_name]
    print(f"  Loading config from {hf_id}...")
    config = build_config(model_name)
    overrides = OVERRIDE_FE_CONFIG.get(model_name, {})
    if overrides:
        print(f"  Applied config overrides: {overrides}")
    model = Wav2Vec2Model(config)

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    print("  Verifying inference...")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 16000))
    print(f"  Output: {out.last_hidden_state.shape}")

    print(f"  Saving to {output_dir}/...")
    model.save_pretrained(str(output_dir))

    full_state = {}
    for k, v in new_state.items():
        full_state[f"m_ssl.model.{k}"] = v
    for k, v in head_keys.items():
        full_state[k] = v
    torch.save(full_state, str(output_dir / "full_model.pt"))

    info = {
        "model_name": model_name,
        "hf_base_model": hf_id,
        "ssl_keys_mapped": mapped,
        "ssl_keys_total": len(ssl_keys),
        "pretraining_keys_skipped": len(skipped),
        "unmapped_keys": len(unmapped),
        "missing_in_model": len(missing),
        "hidden_dim": MODEL_DIMS[model_name],
        "n_layers": MODEL_LAYERS[model_name],
    }
    with open(output_dir / "conversion_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser(description="Convert AntiDeepfake checkpoints to transformers format")
    parser.add_argument("--model", required=True, choices=list(HF_MODEL_IDS.keys()))
    parser.add_argument("--ckpt", type=str, default=None, help="Path to .ckpt file")
    parser.add_argument("--safetensors", type=str, default=None, help="Path to .safetensors file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    if not args.ckpt and not args.safetensors:
        parser.error("Must provide either --ckpt or --safetensors")

    print(f"Converting {args.model}...")
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        src_state = ckpt.get("model", ckpt)
    else:
        from safetensors.torch import load_file
        src_state = load_file(args.safetensors)

    info = convert_checkpoint(args.model, src_state, args.output)
    print(f"\nDone! Mapped {info['ssl_keys_mapped']}/{info['ssl_keys_total']} keys -> {args.output}/")


if __name__ == "__main__":
    main()
