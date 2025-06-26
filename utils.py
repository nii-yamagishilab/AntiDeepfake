"""This script provides a collection of functions used during training pipelines,
including:

* Loading model weights - load_weights()
* Setting random seeds for reproducibility - set_random_seed()
"""
import random

import torch
import numpy as np


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

def load_weights(trg_state, path, func_name_change=lambda x: x):
    """load_weights(trg_state, path)
    Load trained weights to the state_dict() of a torch.module on CPU
    """
    # load to CPU
    loaded_state = torch.load(
        path, 
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )

    if 'model' in loaded_state:
        loaded_state = loaded_state['model']

    # customized loading patterns (provided by SASV baseline code)
    for name, param in loaded_state.items():
        origname = name
        if name not in trg_state:
            name = func_name_change(name)
            if name not in trg_state:
                print("{:s} is not in the model.".format(origname))
                continue

        if trg_state[name].size() != loaded_state[origname].size():
            print("Wrong para. length: {:s}, model: {:s}, loaded: {:s}".format(
                origname, trg_state[name].size(), 
                loaded_state[origname].size()))
            continue
        trg_state[name].copy_(param)
    return

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
