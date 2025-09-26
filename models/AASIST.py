#!/usr/bin/env python
"""
"""
import os
import sys
import torch
from models.aasist import AASIST

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"


global_config = {'default':
                 {'architecture': 'AASIST',
                  'nb_samp': 64600,
                  'first_conv': 128,
                  'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                  'gat_dims': [64, 32],
                  'pool_ratios': [0.5, 0.7, 0.5, 0.5],
                  'temperatures': [2.0, 2.0, 100.0, 100.0]
                  }
                 }


import logging
# W2V has logging basicConfig in fairseq, here we manually do that
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr)


class Model(torch.nn.Module):
    """ Model definition
    Args:
      model_name: string, pass to global configurations to get model specific config
    """
    def __init__(self, config_name = 'default'):
        super(Model, self).__init__()

        try:
            d_args = global_config[config_name]
        except KeyError:
            print("Unknown AASIST config {:s}".format(config_name))
            sys.exit(1)
        self.model = AASIST.Model(d_args)
        self.out_dim = self.model.out_layer.weight.shape[1]
        self.d_args = d_args
        return

    @staticmethod
    def name_map(n):
        # used to load weights
        return f"model.{n}"
        
    def __forward(self, wav):
        pooled_emb, pred = self.model(wav)
        return pred, pooled_emb

    def forward(self, wav):
        return self.__forward(wav)[0]
    
    def get_emb_dim(self):
        return self.out_dim
    
    def analysis(self, wav):
        return self.__forward(wav)
