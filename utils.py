"""This script provides a collection of functions used during training pipelines,
including:

* Loading model weights - load_weights()
* Setting random seeds for reproducibility - set_random_seed()
"""
import random
import os
import itertools
import collections
import pickle

import torch
import numpy as np


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

def load_weights(trg_state, path, func_name_change=lambda x: x):
    """Load trained weights to the state_dict() of a torch.module on CPU
    """
    # load to CPU
    try:
        loaded_state = torch.load(
            path, 
            map_location=lambda storage, loc: storage,
            # set to False for loading fariseq pt models: w2v_small, w2v_large, hubert_xl
            weights_only=True,
        )
    except pickle.UnpicklingError as e:
        loaded_state = torch.load(
            path, 
            map_location=lambda storage, loc: storage,
            # set to False for loading fariseq pt models: w2v_small, w2v_large, hubert_xl
            weights_only=False,
        )        
    except:
        assert 1==0, "Fail to load {:s}".format(path)

    # if it is a fairseq-style checkpoint
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


############
# data IO
############
def pickle_dump(data, file_path):
    """ pickle_dump(data, file_path)
    Dump data into a pickle file
                                                                                                    
    inputs:
      data: python object, data to be dumped
      file_path: str, path to save the pickle file
    """
    try:
        os.mkdir(os.path.dirname(file_path))
    except OSError:
        pass

    with open(file_path, 'wb') as file_ptr:
        pickle.dump(data, file_ptr)
    return

def pickle_load(file_path):
    """ data = pickle_load(file_path)
    Load data from a pickle dump file
                                                                          
    inputs:
      file_path: str, path of the pickle file
                                                                  
    output:
      data: python object
    """
    with open(file_path, 'rb') as file_ptr:
        data = pickle.load(file_ptr)
    return data

