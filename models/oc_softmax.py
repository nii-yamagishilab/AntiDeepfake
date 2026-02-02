"""
one class

One-class learning towards generalized voice spoofing detection
Zhang, You and Jiang, Fei and Duan, Zhiyao
arXiv preprint arXiv:2010.13995
"""

import torch
import torch.nn as nn
from torch.nn import Parameter

class OCAngleLayer(nn.Module):
    """Output layer to produce activation for one-class softmax"""

    def __init__(self, input_features, config={}):
        super(OCAngleLayer, self).__init__()
        in_planes = input_features
        w_posi = config["w_posi"] if 'w_posi' in config else 0.9
        w_nega = config["w_nega"] if "w_nega" in config else 0.2
        alpha = config["alpha"] if 'alpha' in config else 20.0

        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1

        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations
        """
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        if flag_angle_only:
            pos_score = cos_theta
            neg_score = -cos_theta
            # we only need pos_score, but to be compatible w/ evaluation
            # tool, we save a fake negative score
            return torch.concat([neg_score, pos_score], dim=1)
        
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
            return pos_score, neg_score


def OCSoftmaxLoss(logits, target):
    """
    logits: output of OCSoftmaxLayer.
    target: (batch,)
    """
    output = logits[0] * target.view(-1, 1) + logits[1] * (1 - target.view(-1, 1))
    return torch.nn.functional.softplus(output).mean()

