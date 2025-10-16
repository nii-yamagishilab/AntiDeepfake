"""This script is used for performance evaluation on the generated score.csv file

Usage: 
1. to print pooled result using all data from the score file:
    python evaluation.py <score.csv>
2. and optionally, to print pooled & subset result
    python evaluation.py <score.csv> ID_PREFIX_1 ID_PREFIX_2 ...

The score.csv file should look like below. Each [Score] is [Fake logits, Real logits],
and [Lable] is 0 for Groundtruth Fake and 1 for Groundtruth Real

score.csv:
ID,Score,Label
ID_PREFIX_1_wav1_ID,"[-3.7587778568267822, 3.772808074951172]",1
ID_PREFIX_1_wav2_ID,"[3.4324636459350586, -3.5558271408081055]",0
ID_PREFIX_2_wav3_ID,"[-3.8531546592712402, 3.871497631072998]",1
ID_PREFIX_2_wav4_ID,"[3.6327223777770996, -3.7554166316986084]",0
"""
import sys
import ast

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"


# ======================================================
# EER Calculation (from ASVspoof)
# ======================================================
def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size \
                           - (np.arange(1, n_scores + 1) - tar_trial_sums)
    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size)
    )
    # false acceptance rates
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold. 
    """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


# ======================================================
# Helper Functions
# ======================================================
def extract_scores(score_str):
    """Read the given [score_str] column and convert string into a list of two logits.
    """
    try:
        score_list = ast.literal_eval(score_str)
        return score_list if score_list else [None, None]
    except Exception as e:
        print(f"Error parsing score: {e}")
        return [None, None]


def get_prefix(id_str):
    """Extract dataset prefix from ID (text before the first '-')
    """
    parts = id_str.split('-')
    return parts[0] if len(parts) > 1 else None

    
def parse_file(filename, subset_list=None):
    """Read and record results from the SCORE.csv file.
    """
    # Read and preprocess the score file
    df = pd.read_csv(filename)
    df["Logits"] = df["Score"].apply(extract_scores)
    df["Dataset"] = df["ID"].apply(get_prefix)
    # To get pooled metrics for the whole score.csv file
    data = {'Pooled': {'y_true': [], 'logits': []}}
    # Only initialize subset data if user specify a list 
    if subset_list is not None:
        for prefix in subset_list:
            data[prefix] = {'y_true': [], 'logits': []}
    # Read and record each entry
    for _, row in df.iterrows():
        file_id = row["ID"].strip()
        logits = row["Logits"]
        label = int(row["Label"])
        dataset = row["Dataset"]
        # Pooled result of the whole score file
        data['Pooled']['y_true'].append(label)
        data['Pooled']['logits'].append(logits)
        # Append to subset if the extracted prefix is in the list.
# Such list is used to avoid the case, when you only want pooled result, 
# but some of your filenames have '-' and they are accidentally grouped to a subset
        if subset_list is not None and dataset in subset_list:
            data[dataset]['y_true'].append(label)
            data[dataset]['logits'].append(logits)

    return data


def softmax_score(logits):
    """This function uses softmax and return the prob of the positive class.
    """
    logits_tensor = torch.tensor(logits)
    softmax_scores = F.softmax(logits_tensor, dim=1).numpy()
    score_positive = softmax_scores[:, 1]
    return score_positive


def compute_metrics(y_true, logits, pred_threshold=0.5):
    """Compute all required metrics.
    """
    # Get the score of positive class ranged [0, 1]
    score_positive = softmax_score(logits)
    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, score_positive, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # EER
    target_scores = score_positive[np.array(y_true) == 1]
    nontarget_scores = score_positive[np.array(y_true) == 0]
    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)
    # Threshold-based predictions (default 0.5)
    predicted_labels = (score_positive >= pred_threshold).astype(int)
    # Precision, Recall, F1, ACC
    precision = precision_score(y_true, predicted_labels)
    recall = recall_score(y_true, predicted_labels)
    f1 = f1_score(y_true, predicted_labels)
    accuracy = accuracy_score(y_true, predicted_labels)
    # False positive rate, False negative rate
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_labels).ravel()
    fpr_custom = fp / (fp + tn)
    fnr_custom = fn / (fn + tp)

    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr_custom,
        'fnr': fnr_custom
    }


def main(filename, subset_list):
    # Preprocess
    data = parse_file(filename, subset_list)
    # Load scores and calculate metrics
    results = []
    for key in data:
        # Valid prefix contains at least "Pooled", and optional user-given prefix
        if subset_list is not None and key != 'Pooled' and key not in subset_list:
            continue

        if not data[key]['y_true']:
            print(f"No data for {key}")
            continue
        # Ground truth result
        y_true = np.array(data[key]['y_true'])
        # Predictions
        logits = np.array(data[key]['logits'])
        # We use 0.5 for metrics with fixed threshold
        pred_threshold = 0.5
        # Compute all metrics here
        metrics = compute_metrics(y_true, logits, pred_threshold)
        metrics['subset'] = key
        results.append(metrics)

    if not results:
        print("No valid data to display.")
        return

    # Convert to DataFrame for tabular display
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('subset')

    # Optionally reorder columns
    df_results = df_results[[
        'roc_auc', 'accuracy', 'precision', 'recall', 'f1',
        'fpr', 'fnr', 'eer', 'eer_threshold'
    ]]
    
    # Format display
    print("\n===== METRICS SUMMARY =====")
    print(f"For accuracy, precision, recall, f1, fpr and fnr, threshold of real class probablity is {pred_threshold}\n")
    print(df_results.round(4).to_string())

if __name__ == "__main__":
    score_file = sys.argv[1]
    subset_list = sys.argv[2:] if len(sys.argv) > 2 else None
    main(score_file, subset_list)
