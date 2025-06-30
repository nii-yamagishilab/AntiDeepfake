"""This script defines a PyTorch-based neural network for the W2V-based AntiDeepfake models.

Classes:
    * SSLModel: Loads the pre-trained Fairseq SSL model and extracts features from raw audio input.
    * Model: A wrapper model that uses SSLModel for feature extraction and adds a projection layer for binary classification.

global_configs contains configurations for each model, they are used to initialize
SSL architectures
"""
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from models.W2V_configs import global_input_dims, global_configs

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"


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
