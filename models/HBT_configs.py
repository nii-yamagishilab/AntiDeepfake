"""Configuration of fairseq SSL HuBERT models
"""

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

global_input_dims = {'hubert_large_ll60k': 1024,
                     'hubert_xlarge_ll60k': 1280}

# dictionary size (including special tokens)
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
