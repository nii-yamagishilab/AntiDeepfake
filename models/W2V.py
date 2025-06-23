"""This script defines a PyTorch-based neural network for the SSL-AntiDeepfake models.

Classes:
    * SSLModel: Loads the pre-trained Fairseq SSL model and extracts features from raw audio input.
    * Model: A wrapper model that uses SSLModel for feature extraction and adds a projection layer for binary classification.
"""

import torch
import fairseq


class SSLModel(torch.nn.Module):
    def __init__(self, cp_path, ssl_orig_output_dim):
        """ SSLModel(cp_path, ssl_orig_output_dim)
        
        Args:
          cp_path: string, path to the pre-trained fairseq SSL model
          ssl_orig_output_dim: int, dimension of the SSL model output feature
        """
        super(SSLModel, self).__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        # dimension of output from SSL model. This is fixed
        self.out_dim = ssl_orig_output_dim
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
        ssl_orig_output_dim: same in SSLModel(cp_path, ssl_orig_output_dim)
        ssl_path: cp_path in SSLModel(cp_path, ssl_orig_output_dim)
    """
    def __init__(self, ssl_orig_output_dim, ssl_path):
        super(Model, self).__init__()
        # number of output class
        self.v_out_class = 2
        
        ####
        # create network
        ####
        self.m_ssl = SSLModel(ssl_path, ssl_orig_output_dim)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(
            output_size=1
        )
        self.proj_fc = torch.nn.Linear(
            in_features=ssl_orig_output_dim,
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
