"""This script defines a PyTorch-based neural network for the W2V-based AntiDeepfake models.

Classes:
    * SSLModel: Loads the pre-trained Fairseq SSL model and extracts features from raw audio input.
    * Model: A wrapper model that uses SSLModel for feature extraction and adds a projection layer for binary classification.

global_configs contains configurations for each model, they are used to initialize
SSL architectures
"""
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

global_input_dims = {
    'w2v_small': 768, 'w2v_large': 1024, 'mms_300m': 1024, 'mms_1b': 1280, 'xlsr_1b': 1280, 'xlsr_2b': 1920,
}

global_configs = {
    'w2v_small': {'_name': 'wav2vec2', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'final_dim': 256, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'feature_grad_mult': 0.1, 'quantize_targets': True, 'latent_temp': [2.0, 0.5, 0.999995]},
    'w2v_large': {'_name': 'wav2vec2', 'extractor_mode': 'layer_norm', 'encoder_layers': 24, 'encoder_embed_dim': 1024, 'encoder_ffn_embed_dim': 4096, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 768, 'layer_norm_first': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': True, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.1, 0.999995]},
    'mms_300m': {'_name': 'wav2vec2', 'extractor_mode': 'layer_norm', 'encoder_layers': 24, 'encoder_embed_dim': 1024, 'encoder_ffn_embed_dim': 4096, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 768, 'layer_norm_first': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': True, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.1, 0.999995]},
    'mms_1b': {'_name': 'wav2vec2', 'extractor_mode': 'layer_norm', 'encoder_layers': 48, 'encoder_embed_dim': 1280, 'encoder_ffn_embed_dim': 5120, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 1024, 'layer_norm_first': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': True, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.1, 0.999995]},
    'xlsr_1b': {'_name': 'wav2vec2', 'extractor_mode': 'layer_norm', 'encoder_layers': 48, 'encoder_embed_dim': 1280, 'encoder_ffn_embed_dim': 5120, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 1024, 'layer_norm_first': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': True, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.1, 0.999995]},
    'xlsr_2b': {'_name': 'wav2vec2', 'extractor_mode': 'layer_norm', 'encoder_layers': 48, 'encoder_embed_dim': 1920, 'encoder_ffn_embed_dim': 7680, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 1024, 'layer_norm_first': True, 'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]', 'conv_bias': True, 'logit_temp': 0.1, 'quantize_targets': True, 'quantize_input': False, 'same_quantizer': False, 'target_glu': False, 'feature_grad_mult': 1.0, 'quantizer_depth': 1, 'quantizer_factor': 3, 'latent_vars': 320, 'latent_groups': 2, 'latent_dim': 0, 'mask_length': 10, 'mask_prob': 0.65, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_before': False, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'num_negatives': 100, 'negatives_from_everywhere': False, 'cross_sample_negatives': 0, 'codebook_negatives': 0, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.1, 0.999995]}
}

class SSLModel(torch.nn.Module):
    def __init__(self, model_name):
        """ SSLModel(model_name)
        Args:
          model_name: string, pass to global configurations to get model specific config
        """
        super(SSLModel, self).__init__()
        # model-specific config 
        config = global_configs[model_name]
        cfg = Wav2Vec2Config(**config)
        # randomly initialized model
        self.model = Wav2Vec2Model(cfg)
        # dimension of output from SSL model
        self.out_dim = global_input_dims[model_name]
        return
    
    def extract_feat(self, input_data):
        """ feature = extract_feat(input_data)
        Args:
          input_data: tensor, waveform, (batch, T)
        
        Return:
          feature: tensor, feature, (batch, frame, hidden_dim)
        """
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            #self.model.eval()
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

class Model(torch.nn.Module):
    """ Model definition
    Args:
      model_name: string, pass to global configurations to get model specific config
    """
    def __init__(self, model_name):
        super(Model, self).__init__()
        # number of output class
        self.v_out_class = 2
        
        ####
        # create network
        ####
        self.m_ssl = SSLModel(model_name)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(
            output_size=1
        )
        self.proj_fc = torch.nn.Linear(
            in_features=self.m_ssl.out_dim,
            out_features=self.v_out_class,
        )
        
    def forward(self, wav):
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
        return pred
