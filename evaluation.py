"""This script is used for performance evaluation on the generated score.csv file

Usage: python evaluation.py <score.csv>

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

def compute_eer(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr 
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold, fpr, tpr, thresholds

def extract_scores(score_str):
    try:
        score_list = ast.literal_eval(score_str)
        return score_list if score_list else [None, None]
    except Exception as e:
        print(f"Error parsing score: {e}")
        return [None, None]

def parse_file(filename):
    data = {
        # To get pooled metrics for the whole score.csv file
        'all': {'y_true': [], 'logits': []},
        # To get metrics for each specific databases,
        # based on the ID_PREFIX defined when saving database protocol
        'ID_PREFIX_1': {'y_true': [], 'logits': []},
        'ID_PREFIX_2': {'y_true': [], 'logits': []},
        }

    df = pd.read_csv(filename)
    df["Logits"] = df["Score"].apply(extract_scores)

    for _, row in df.iterrows():
        file_id = row["ID"].strip()
        logits = row["Logits"]
        label = int(row["Label"])

        if logits is None or len(logits) < 2:
            continue

        for key in data:
            if key == 'all' or file_id.startswith(key):
                data[key]['y_true'].append(label)
                data[key]['logits'].append(logits[:2])  # Only first two logits used

    return data

def compute_metrics(y_true, logits, pred_threshold=0.5):
    logits_tensor = torch.tensor(logits)
    softmax_scores = F.softmax(logits_tensor, dim=1).numpy()
    score_positive = softmax_scores[:, 1]

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(y_true, score_positive, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # EER
    eer, eer_threshold, _, _, _ = compute_eer(y_true, score_positive)

    # Threshold-based predictions (default 0.5)
    predicted_labels = (score_positive >= pred_threshold).astype(int)

    precision = precision_score(y_true, predicted_labels)
    recall = recall_score(y_true, predicted_labels)
    f1 = f1_score(y_true, predicted_labels)
    accuracy = accuracy_score(y_true, predicted_labels)
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

def main(filename):
    data = parse_file(filename)
    results = []

    for key in data:
        if not data[key]['y_true']:
            print(f"No data for {key}")
            continue

        y_true = np.array(data[key]['y_true'])
        logits = np.array(data[key]['logits'])

        pred_threshold = 0.5

        metrics = compute_metrics(y_true, logits, pred_threshold)
        metrics['subset'] = key  # Add subset name to metrics
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
    if len(sys.argv) != 2:
        print("Usage: python evaluation.py <score_file>")
        sys.exit(1)

    score_file = sys.argv[1]
    main(score_file)
