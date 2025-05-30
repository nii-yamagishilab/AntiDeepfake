"""This script provides a collection of functions used during training pipelines,
including:

* Loading model weights - load_weights()
* Setting random seeds for reproducibility - set_random_seed()
* Computing Equal Error Rate (EER) for binary classification tasks - compute_eer()
* RawBoost data augmentation - process_Rawboost_feature(), reproduced based on https://github.com/TakHemlata/RawBoost-antispoofing 
"""

import random
import copy

import torch
import numpy as np
from scipy import signal


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

def load_weights(trg_state, path, func_name_change=lambda x: x): 
    """load_weights(trg_state, path)
    Load trained weights to the state_dict() of a torch.module on CPU
    
    """
    # load to CPU
    loaded_state = torch.load(path, map_location=lambda storage, loc: storage)

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

# def compute_eer(target_scores, nontarget_scores):
def compute_eer(label_list, score_list):
    label_list = np.array(label_list)
    score_list = np.array(score_list)
    target_scores = score_list[label_list == 1]
    nontarget_scores = score_list[label_list == 0]
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


### Algorithms used for RawBoost ###
def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x,always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x


def genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    if x[p].dtype != r.dtype:
        r = r.astype(x[p].dtype)
    y[p] = x[p] + r
    y = normWav(y,0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x


### Actual RawBoost data augmentation function ###
def process_Rawboost_feature(feature, sr, args, algo):
    original_feature_dtype = feature.dtype
    feature = feature.numpy()
    if algo == 1:
        params = args['LnL_convolutive_noise']
        feature = LnL_convolutive_noise(
            feature,
            params['N_f'],
            params['nBands'],
            params['minF'],
            params['maxF'],
            params['minBW'],
            params['maxBW'],
            params['minCoeff'],
            params['maxCoeff'],
            params['minG'],
            params['maxG'],
            params['minBiasLinNonLin'],
            params['maxBiasLinNonLin'],
            sr
        )
    elif algo == 2:
        params = args['ISD_additive_noise']
        feature = ISD_additive_noise(
            feature,
            params['P'],
            params['g_sd']
        )
    elif algo == 3:
        params = args['SSI_additive_noise']
        feature = SSI_additive_noise(
            feature,
            params['SNRmin'],
            params['SNRmax'],
            args['LnL_convolutive_noise']['nBands'],
            args['LnL_convolutive_noise']['minF'],
            args['LnL_convolutive_noise']['maxF'],
            args['LnL_convolutive_noise']['minBW'],
            args['LnL_convolutive_noise']['maxBW'],
            args['LnL_convolutive_noise']['minCoeff'],
            args['LnL_convolutive_noise']['maxCoeff'],
            args['LnL_convolutive_noise']['minG'],
            args['LnL_convolutive_noise']['maxG'],
            sr
        )
    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature,
            args['LnL_convolutive_noise']['N_f'],
            args['LnL_convolutive_noise']['nBands'],
            args['LnL_convolutive_noise']['minF'],
            args['LnL_convolutive_noise']['maxF'],
            args['LnL_convolutive_noise']['minBW'],
            args['LnL_convolutive_noise']['maxBW'],
            args['LnL_convolutive_noise']['minCoeff'],
            args['LnL_convolutive_noise']['maxCoeff'],
            args['LnL_convolutive_noise']['minG'],
            args['LnL_convolutive_noise']['maxG'],
            args['LnL_convolutive_noise']['minBiasLinNonLin'],
            args['LnL_convolutive_noise']['maxBiasLinNonLin'],
            sr
        )
        feature = ISD_additive_noise(
            feature,
            args['ISD_additive_noise']['P'],
            args['ISD_additive_noise']['g_sd']
        )
        feature = SSI_additive_noise(
            feature,
            args['SSI_additive_noise']['SNRmin'],
            args['SSI_additive_noise']['SNRmax'],
            args['LnL_convolutive_noise']['nBands'],                                    
            args['LnL_convolutive_noise']['minF'],                                      
            args['LnL_convolutive_noise']['maxF'],                                      
            args['LnL_convolutive_noise']['minBW'],                                     
            args['LnL_convolutive_noise']['maxBW'],                                     
            args['LnL_convolutive_noise']['minCoeff'],                                  
            args['LnL_convolutive_noise']['maxCoeff'],                                  
            args['LnL_convolutive_noise']['minG'],                                      
            args['LnL_convolutive_noise']['maxG'],                                      
            sr                                                                          
        )
    elif algo == 5:
        feature = LnL_convolutive_noise(                                                
            feature,                                                                    
            args['LnL_convolutive_noise']['N_f'],                                       
            args['LnL_convolutive_noise']['nBands'],                                    
            args['LnL_convolutive_noise']['minF'],                                      
            args['LnL_convolutive_noise']['maxF'],                                      
            args['LnL_convolutive_noise']['minBW'],                                     
            args['LnL_convolutive_noise']['maxBW'],                                     
            args['LnL_convolutive_noise']['minCoeff'],                                  
            args['LnL_convolutive_noise']['maxCoeff'],                                  
            args['LnL_convolutive_noise']['minG'],                                      
            args['LnL_convolutive_noise']['maxG'],                                      
            args['LnL_convolutive_noise']['minBiasLinNonLin'],                          
            args['LnL_convolutive_noise']['maxBiasLinNonLin'],                          
            sr                                                                          
        )
        feature = ISD_additive_noise(                                                   
            feature,                                                                    
            args['ISD_additive_noise']['P'],                                            
            args['ISD_additive_noise']['g_sd']                                          
        )
    elif algo == 6:
        feature = LnL_convolutive_noise(                                                
            feature,                                                                    
            args['LnL_convolutive_noise']['N_f'],                                       
            args['LnL_convolutive_noise']['nBands'],                                    
            args['LnL_convolutive_noise']['minF'],                                      
            args['LnL_convolutive_noise']['maxF'],                                      
            args['LnL_convolutive_noise']['minBW'],                                     
            args['LnL_convolutive_noise']['maxBW'],                                     
            args['LnL_convolutive_noise']['minCoeff'],                                  
            args['LnL_convolutive_noise']['maxCoeff'],                                  
            args['LnL_convolutive_noise']['minG'],                                      
            args['LnL_convolutive_noise']['maxG'],                                      
            args['LnL_convolutive_noise']['minBiasLinNonLin'],                          
            args['LnL_convolutive_noise']['maxBiasLinNonLin'],                          
            sr                                                                          
        )
        feature = SSI_additive_noise(
            feature,
            args['SSI_additive_noise']['SNRmin'],
            args['SSI_additive_noise']['SNRmax'],
            args['LnL_convolutive_noise']['nBands'],
            args['LnL_convolutive_noise']['minF'],
            args['LnL_convolutive_noise']['maxF'],
            args['LnL_convolutive_noise']['minBW'],
            args['LnL_convolutive_noise']['maxBW'],
            args['LnL_convolutive_noise']['minCoeff'],
            args['LnL_convolutive_noise']['maxCoeff'],
            args['LnL_convolutive_noise']['minG'],                                      
            args['LnL_convolutive_noise']['maxG'],                                      
            sr                                                                          
        )
    elif algo == 7:
        feature = ISD_additive_noise(                                                   
            feature,                                                                    
            args['ISD_additive_noise']['P'],                                            
            args['ISD_additive_noise']['g_sd']                                          
        )
        feature = SSI_additive_noise(                                                   
            feature,                                                                    
            args['SSI_additive_noise']['SNRmin'],                                     
            args['SSI_additive_noise']['SNRmax'],                                     
            args['LnL_convolutive_noise']['nBands'],                                    
            args['LnL_convolutive_noise']['minF'],                                      
            args['LnL_convolutive_noise']['maxF'],                                      
            args['LnL_convolutive_noise']['minBW'],                                     
            args['LnL_convolutive_noise']['maxBW'],                                     
            args['LnL_convolutive_noise']['minCoeff'],                                  
            args['LnL_convolutive_noise']['maxCoeff'],                                  
            args['LnL_convolutive_noise']['minG'],                                      
            args['LnL_convolutive_noise']['maxG'],                                      
            sr                                                                          
        )
    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature,
            args['LnL_convolutive_noise']['N_f'],
            args['LnL_convolutive_noise']['nBands'],
            args['LnL_convolutive_noise']['minF'],
            args['LnL_convolutive_noise']['maxF'],
            args['LnL_convolutive_noise']['minBW'],
            args['LnL_convolutive_noise']['maxBW'],
            args['LnL_convolutive_noise']['minCoeff'],
            args['LnL_convolutive_noise']['maxCoeff'],
            args['LnL_convolutive_noise']['minG'],
            args['LnL_convolutive_noise']['maxG'],
            args['LnL_convolutive_noise']['minBiasLinNonLin'],
            args['LnL_convolutive_noise']['maxBiasLinNonLin'],
            sr
        )
        feature2 = ISD_additive_noise(
            feature,
            args['ISD_additive_noise']['P'],
            args['ISD_additive_noise']['g_sd']
        )
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    else:
        feature = feature
    # Convert back to torch Tensor
    if not isinstance(feature, torch.Tensor):
        feature = torch.tensor(feature)
    # Check if RawBoost changes dtype and keep it same as original before processing
    if feature.dtype != original_feature_dtype:
        feature = feature.to(original_feature_dtype)
    return feature
