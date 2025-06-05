"""
This script generates prediction scores from a specific model on a specified dataset,
without using `main.py evaluate()`.

Usage:
    python test.py

Before running the script, please follow the steps below to modify the code accordingly:

Step [1]: Specify the actual path to your own checkpoint.
    - This should be the path to the `.ckpt` file saved when you ran `main.py`.
    - Default path format:
        <base_path>/Log/exps/exp_mms_1b_exp/save/CKPT+2025-01-23+09-40-59+00/ssl.ckpt

    Also, specify the path to the Fairseq checkpoint used to build the SSL front-end.
    - Default path format:
        <base_path>/Log/ssl-weights/base_1b.pt

    Additionally, set the output feature dimension for the SSL front-end.
    - set ssl_orig_output_dim to 768 for w2v_small
    - 1024 for w2v_large, mms_300m
    - 1280 for mms_1b, xlsr_1b, hubert_xl
    - 1920 for xlsr_2b 

Step [2]: Provide the path to your test dataset and its corresponding protocol file.
    - You can refer to any .py files in ./protocols to generate corresponding protocols for your data
    - Any protocol file works if it has the three required columns:
        [ID] [Path] [Label]
    
"""
import csv

import torch
import torchaudio
import pandas as pd

from models.W2V import Model


__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

def load_and_preprocess(wav_path):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)
    wav = torchaudio.functional.resample(wav, sr, new_freq=16000,)
    with torch.no_grad():
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    return wav.unsqueeze(0).cuda()


# Step[0]: Name of the output score .csv file
score_name = "SCORE"

# Step[1]: modify here for each used SSL models
ssl_orig_output_dim = 1280
ckpt_path = "/path/to/your/own/checkpoint/ssl.ckpt"
ssl_path = "/path/to/fairseq/model/checkpoint/base_1b.pt"

# Step[2]: Define paths for loading audio files
# Your database_protocol.csv should have a [Path] column, which reads like:
# $ROOT/SOME_DATASET/TEST/IT/PIZZA/QUATTRO/STAGIONI.wav
# in this case, modify root_path so that each audio is loaded by its absolute path:
# /path/to/your/SOME_DATASET/TEST/IT/PIZZA/QUATTRO/STAGIONI.wav
root_path = '/path/to/your/'
protocol = '/path/to/your/evaluation/protocol/database_protocol.csv'
protocol_df = pd.read_csv(protocol)
protocol_df["Path"] = protocol_df["Path"].str.replace("$ROOT/", root_path)

# Load SSL protocols
ssl = Model(ssl_orig_output_dim, ssl_path)
state_dict = torch.load(ckpt_path, weights_only=True)
ssl.load_state_dict(state_dict)
ssl.cuda().eval()

# Score generation 
with torch.no_grad():
    id_list = []
    score_list = []
    label_list = []
    for index, row in protocol_df.iterrows():
        # Skip very short audio
#         audio_len = float(row["Duration"])
#         if audio_len < 0.5:
#             continue
        wav_path = row["Path"]
        file_id = row["ID"]
        wav = load_and_preprocess(wav_path)
        label = row["Label"] 
        if label == 'real':
            label = 1
        else:
            label = 0
        prediction = ssl.forward(wav)
        scores = prediction.cpu().data.numpy().flatten().tolist()

        id_list.append(file_id)
        score_list.append(scores)
        label_list.append(label)

    # Save CSV score
    csv_path = f'{score_name}.csv'
    with open(csv_path, mode='w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ID', 'Score', 'Label'])
        for file_id, scores, label in zip(id_list, score_list, label_list):
            writer.writerow([file_id, str(scores), label])
    print("CSV file saved")
