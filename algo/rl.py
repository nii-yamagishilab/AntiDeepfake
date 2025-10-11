"""This script is used for model training.
"""
import os
import sys
import random

import numpy as np

import torch
import torch.nn.functional as F
from speechbrain.utils.logger import get_logger

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

logger = get_logger(__name__)

def grpo_loss(logits, logits_ref, sa_labels, gt_labels, rl_config):
    """compute grpo loss for classifier
    input: logits, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: logits_ref, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: sa_labels, tensor, (B, M), model sampled labels, batch size B, M samples for each data
    input: gt_labels, tensor, (B), groud-truth labels
    input: rl_config, dict, configuration


    Wrong implementation. Leave here for record
    """
    # sample num
    samp_num = sa_labels.shape[-1]
    
    # reshape tensors
    # duplicate logits (B, L) -> (B x M, L)
    logits_ = torch.repeat_interleave(logits, samp_num, dim=0)
    logits_ref_ = torch.repeat_interleave(logits_ref, samp_num, dim=0)
    
    # duplicate gt_labels, (B, ) -> (BxM, )
    gt_labels_ = torch.repeat_interleave(gt_labels, samp_num)
    # reshape sa-labels (B, M) -> (BxM, )
    sa_labels_ = torch.flatten(sa_labels)


    # compute reward
    # assuming a simple indicator loss
    reward = torch.zeros_like(gt_labels_, dtype=logits.dtype)
    reward[gt_labels_ == sa_labels_] = 1.0
    reward[gt_labels_ != sa_labels_] = 0.0
    
    # norm the reward
    reward_norm = (reward - reward.mean())/(reward.std() + rl_config['std_floor'])
    
    # compute weighted cross entropy loss
    loss_raw = torch.nn.functional.cross_entropy(logits_, sa_labels_, reduction='none')

    # KLD 
    # log_softmax
    # select the logits based on the labels
    logps = torch.nn.functional.log_softmax(logits_, dim=-1)
    logps_ref = torch.nn.functional.log_softmax(logits_ref_, dim=-1)
    
    logps_ = torch.gather(logps, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    logps_ref_ = torch.gather(logps_ref, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    
    kl = torch.exp(logps_ref_ - logps_) - (logps_ref_ - logps_) - 1
    kl_beta = rl_config['beta'] if 'beta' in rl_config else 1.0
    
    # weighted by reward
    loss = (loss_raw * reward_norm - kl_beta * kl).mean() * -1.0

    return loss


def grpo2_loss(logits, logits_ref, sa_labels, gt_labels, rl_config):
    """compute grpo loss for classifier
    input: logits, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: logits_ref, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: sa_labels, tensor, (B, M), model sampled labels, batch size B, M samples for each data
    input: gt_labels, tensor, (B), groud-truth labels
    input: rl_config, dict, configuration
    """
    # sample num
    samp_num = sa_labels.shape[-1]
    
    # reshape tensors
    # duplicate logits (B, L) -> (B x M, L)
    logits_ = torch.repeat_interleave(logits, samp_num, dim=0)
    logits_ref_ = torch.repeat_interleave(logits_ref, samp_num, dim=0)
    
    # duplicate gt_labels, (B, ) -> (BxM, )
    gt_labels_ = torch.repeat_interleave(gt_labels, samp_num)
    # reshape sa-labels (B, M) -> (BxM, )
    sa_labels_ = torch.flatten(sa_labels)


    # compute reward
    # assuming a simple indicator loss
    reward = torch.zeros_like(gt_labels_, dtype=logits.dtype)
    reward[gt_labels_ == sa_labels_] = 1.0
    reward[gt_labels_ != sa_labels_] = 0.0
    
    # norm the reward
    reward_norm = (reward - reward.mean())/(reward.std() + rl_config['std_floor'])
    
    # compute weighted cross entropy loss
    # ce = - gather(log_softmax(logits), target_labels)
    loss_raw = torch.nn.functional.cross_entropy(logits_, sa_labels_, reduction='none')
    # the mean of log prog = -ce
    # note that logps for KL is the vector of probability for both real/fake
    # here, the logp is the probability P(y=sa_label)    
    logp = -loss_raw 
    
    # KLD 
    # log_softmax
    # select the logits based on the labels
    logps = torch.nn.functional.log_softmax(logits_, dim=-1)
    logps_ref = torch.nn.functional.log_softmax(logits_ref_, dim=-1)
    
    logps_ = torch.gather(logps, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    logps_ref_ = torch.gather(logps_ref, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    
    kl = torch.exp(logps_ref_ - logps_) - (logps_ref_ - logps_) - 1
    kl_beta = rl_config['beta'] if 'beta' in rl_config else 1.0
    
    # weighted by reward
    # following the https://huggingface.co/docs/trl/main/en/grpo_trainer
    # and https://github.com/lsdefine/simple_GRPO/blob/main/simple_grpo_v1/grpo_ref_split.py#L192
    reward = (torch.exp((logp - logp.detach())) * reward_norm - kl_beta * kl).mean()

    return -reward


def grpo3_loss(logits, logits_ref, logits_old, sa_labels, gt_labels, rl_config):
    """compute grpo loss for classifier
    input: logits, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: logits_ref, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: logits_old, tensor, (B, L), model output logits, batch size B, L output dimensions
    input: sa_labels, tensor, (B, M), model sampled labels, batch size B, M samples for each data
    input: gt_labels, tensor, (B), groud-truth labels
    input: rl_config, dict, configuration
    """
    # sample num
    samp_num = sa_labels.shape[-1]
    
    # reshape tensors
    # duplicate logits (B, L) -> (B x M, L)
    logits_ = torch.repeat_interleave(logits, samp_num, dim=0)
    logits_ref_ = torch.repeat_interleave(logits_ref, samp_num, dim=0)
    logits_old_ = torch.repeat_interleave(logits_old, samp_num, dim=0)
    
    # duplicate gt_labels, (B, ) -> (BxM, )
    gt_labels_ = torch.repeat_interleave(gt_labels, samp_num)
    # reshape sa-labels (B, M) -> (BxM, )
    sa_labels_ = torch.flatten(sa_labels)

    # compute reward
    # assuming a simple indicator loss
    reward = torch.zeros_like(gt_labels_, dtype=logits.dtype)
    reward[gt_labels_ == sa_labels_] = 1.0
    reward[gt_labels_ != sa_labels_] = 0.0
    
    # norm the reward
    reward_norm = (reward - reward.mean())/(reward.std() + rl_config['std_floor'])
    
    # compute weighted cross entropy loss
    # ce = - gather(log_softmax(logits), target_labels)
    loss_raw = torch.nn.functional.cross_entropy(logits_, sa_labels_, reduction='none')
    # the mean of log prog = -ce
    # note that logps for KL is the vector of probability for both real/fake
    # here, the logp is the probability P(y=sa_label)    
    logp = -loss_raw 

    loss_raw = torch.nn.functional.cross_entropy(logits_old_, sa_labels_, reduction='none')
    logp_old = -loss_raw 

    # ratio between the two
    r_ = torch.exp(logp - logp_old)
    r_clipped = torch.clamp(r_, 1.0 - rl_config['epsilon'], 1.0 + rl_config['epsilon'])
    
    # KLD 
    # log_softmax
    # select the logits based on the labels
    logps = torch.nn.functional.log_softmax(logits_, dim=-1)
    logps_ref = torch.nn.functional.log_softmax(logits_ref_, dim=-1)
    
    logps_ = torch.gather(logps, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    logps_ref_ = torch.gather(logps_ref, dim=1, index=sa_labels_.unsqueeze(1)).squeeze(1)
    
    kl = torch.exp(logps_ref_ - logps_) - (logps_ref_ - logps_) - 1
    kl_beta = rl_config['beta'] if 'beta' in rl_config else 1.0
    
    # weighted by reward
    # following the https://huggingface.co/docs/trl/main/en/grpo_trainer
    # and https://github.com/lsdefine/simple_GRPO/blob/main/simple_grpo_v1/grpo_ref_split.py#L189
    reward = (torch.min(r_ * reward_norm, r_clipped * reward_norm)  - kl_beta * kl).mean()

    # minimize the loss
    return -reward


