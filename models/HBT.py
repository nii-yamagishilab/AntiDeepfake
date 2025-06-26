"""This script defines a PyTorch-based neural network for Hubert-based AntiDeepfake models.

Classes:
    * SSLModel: Loads the pre-trained Fairseq SSL model and extracts features from raw audio input.
    * Model: A wrapper model that uses SSLModel for feature extraction and adds a projection layer for binary classification.

<global_configs> contains configurations for each model, they are used to initialize
SSL architectures
"""
import torch
from fairseq.models.hubert import HubertModel, HubertConfig
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.data import Dictionary


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

global_input_dims = {'hubert_large_ll60k': 1024,
                     'hubert_xlarge_ll60k': 1280}

global_dicts = {'hubert_large_ll60k': 504,
                'hubert_xlarge_ll60k': 504}

global_tasks = {
    'hubert_large_ll60k': {'_name': 'hubert_pretraining', 'data': '/checkpoint/wnhsu/data/librivox', 'labels': ['lyr9.km500'], 'label_dir': '/checkpoint/wnhsu/experiments/hubert/kmeans_20210121/km_dataset_librivox.model_iter_2.all', 'label_rate': 50, 'sample_rate': 16000, 'normalize': True, 'enable_padding': False, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False},
    'hubert_xlarge_ll60k': {'_name': 'hubert_pretraining', 'data': '/checkpoint/wnhsu/data/librivox', 'labels': ['lyr9.km500'], 'label_dir': '/checkpoint/wnhsu/experiments/hubert/kmeans_20210121/km_dataset_librivox.model_iter_2.all', 'label_rate': 50, 'sample_rate': 16000, 'normalize': True, 'enable_padding': False, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False}
}
global_configs = {
    'hubert_large_ll60k': {'_name': 'hubert', 'label_rate': 50, 'extractor_mode': 'layer_norm', 'encoder_layers': 24, 'encoder_embed_dim': 1024, 'encoder_ffn_embed_dim': 4096, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 768, 'untie_final_proj': True, 'layer_norm_first': True, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 1.0, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': True},
    'hubert_xlarge_ll60k': {'_name': 'hubert', 'label_rate': 50, 'extractor_mode': 'layer_norm', 'encoder_layers': 48, 'encoder_embed_dim': 1280, 'encoder_ffn_embed_dim': 5120, 'encoder_attention_heads': 16, 'activation_fn': 'gelu', 'dropout': 0.0, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.0, 'dropout_input': 0.0, 'dropout_features': 0.0, 'final_dim': 1024, 'untie_final_proj': True, 'layer_norm_first': True, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 1.0, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': True}
}

class SSLModel(torch.nn.Module):
    def __init__(self, model_name):
        """ SSLModel(model_name)
        Args:
          model_name: string, path to global configurations to get model specific config
        """
        super(SSLModel, self).__init__()
        # model-specific config 
        cfg = HubertConfig(**global_configs[model_name])
        # task, which will provide the sampling rate
        task = HubertPretrainingConfig(**global_tasks[model_name])
        # size of dictionary
        dict_size = global_dicts[model_name]
        # create a dummy dictionary
        dict_hubert = Dictionary()
        for i in range(dict_size-len(dict_hubert)):
            _ = dict_hubert.add_symbol(i)
        # randomly initialized model
        self.model = HubertModel(cfg, task, [dict_hubert])
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
      model_name: string, path to global configurations to get model specific config
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
            in_features=global_input_dims[model_name] ,
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
