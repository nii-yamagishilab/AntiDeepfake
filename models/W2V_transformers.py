"""Native HuggingFace Transformers backend for W2V-based AntiDeepfake models.

Drop-in replacement for W2V.py that uses transformers.Wav2Vec2Model instead of
fairseq, enabling use with Python 3.10+ without building fairseq from source.

Classes:
    * SSLModel: Loads the SSL model via HuggingFace transformers and extracts features.
    * Model: Same wrapper as W2V.py with AdaptiveAvgPool1d + Linear classifier.

Usage:
    # Same interface as W2V.Model
    from models.W2V_transformers import Model
    model = Model('mms_300m')

Requires:
    pip install transformers safetensors

Weight conversion from fairseq format:
    python convert_to_transformers.py --model mms_300m --ckpt path/to/mms_300m.ckpt
"""
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config
from models.W2V_configs import global_input_dims

__author__ = "DOS AI (port), Wanying Ge, Xin Wang (original)"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Map model names to HuggingFace model IDs (for config loading)
HF_MODEL_IDS = {
    'w2v_small': 'facebook/wav2vec2-base',
    'w2v_large': 'facebook/wav2vec2-large',
    'mms_300m': 'facebook/mms-300m',
    'mms_1b': 'facebook/mms-1b-all',
    'xlsr_1b': 'facebook/wav2vec2-xls-r-1b',
    'xlsr_2b': 'facebook/wav2vec2-xls-r-2b',
}

# Per-model feature-extractor config overrides. Must stay in sync with
# convert_to_transformers.py: the NII AntiDeepfake fairseq checkpoints use
# feat_extract_norm="layer" + conv_bias=True, but the HF bases for
# wav2vec2-base and wav2vec2-large default to "group" + False and need
# explicit overrides so the built Wav2Vec2Model architecture matches the
# converted state_dict. mms-1b-all is intentionally omitted: its HF default
# is already "layer" + True; the separate mms_1b failure is caused by
# adapter layers in that ASR fine-tune and is tracked separately.
OVERRIDE_FE_CONFIG = {
    'w2v_small': {'feat_extract_norm': 'layer', 'conv_bias': True},
    'w2v_large': {'feat_extract_norm': 'layer', 'conv_bias': True},
}


class SSLModel(torch.nn.Module):
    def __init__(self, model_name):
        """SSLModel(model_name)
        Args:
          model_name: string, one of: w2v_small, w2v_large, mms_300m, mms_1b, xlsr_1b, xlsr_2b
        """
        super(SSLModel, self).__init__()
        hf_id = HF_MODEL_IDS[model_name]
        config = Wav2Vec2Config.from_pretrained(hf_id)
        for k, v in OVERRIDE_FE_CONFIG.get(model_name, {}).items():
            setattr(config, k, v)
        # Create model structure with empty weights
        self.model = Wav2Vec2Model(config)
        self.out_dim = global_input_dims[model_name]
        return

    def extract_feat(self, input_data):
        """feature = extract_feat(input_data)
        Args:
          input_data: tensor, waveform, (batch, T)

        Return:
          feature: tensor, feature, (batch, frame, hidden_dim)
        """
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        # [batch, length, dim]
        out = self.model(input_tmp)
        return out.last_hidden_state


class Model(torch.nn.Module):
    """Model definition - identical interface to W2V.Model
    Args:
      model_name: string, one of the supported model names
    """
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.v_out_class = 2

        self.m_ssl = SSLModel(model_name)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(
            output_size=1
        )
        self.proj_fc = torch.nn.Linear(
            in_features=self.m_ssl.out_dim,
            out_features=self.v_out_class,
        )

    def __forward(self, wav):
        # [batch, frame, hidden_dim]
        emb = self.m_ssl.extract_feat(wav)
        # [batch, hidden_dim, frame]
        emb = emb.transpose(1, 2)
        # [batch, hidden_dim, 1]
        pooled_emb = self.adap_pool1d(emb)
        # [batch, hidden_dim]
        pooled_emb = pooled_emb.view(emb.size(0), -1)
        # [batch, 2]
        pred = self.proj_fc(pooled_emb)
        return pred, pooled_emb

    def forward(self, wav):
        return self.__forward(wav)[0]

    def get_emb_dim(self):
        return self.m_ssl.out_dim

    def analysis(self, wav):
        return self.__forward(wav)
