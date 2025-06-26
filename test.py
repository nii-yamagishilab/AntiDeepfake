"""
This script generates prediction scores from a specific model on a specified dataset,
without using `main.py evaluate()`.

Usage:
    python test.py

Before running the script, please follow the steps below to modify the code accordingly:

Step [1]: Specify model name and the actual path to your own checkpoint.
    - model names: w2v_small; w2v_large; mms_300m; mms_1b; xlsr_1b; xlsr_2b;
                   hubert_xlarge_ll60k;
    - ckpt_path should be the path to the `.ckpt` file saved when you ran `main.py`.
    - Default path format:
        <base_path>/Log/exps/exp_mms_1b_exp/save/CKPT+2025-01-23+09-40-59+00/ssl.ckpt

Step [2]: Provide the path to your test dataset and its corresponding protocol file.
    - You can refer to any .py files in ./protocols to generate corresponding protocols for your data
    - Any protocol file works if it has the three required columns:
        [ID] [Path] [Label]
    
"""
import csv

import torch
import torchaudio
import pandas as pd

# use W2V for wav2vec2 and HBT for Hubert
from models.W2V import Model
# from models.HBT import Model


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
model_name = 'mms_300m'
ckpt_path = "/path/to/your/own/checkpoint/ssl.ckpt"

# Step[2]: Define paths for loading audio files
# replace "$ROOT/" with root_path so that each audio is loaded by its absolute path
root_path = '/path/to/your/'
protocol = '/path/to/your/evaluation/protocol/database_protocol.csv'
protocol_df = pd.read_csv(protocol)
protocol_df["Path"] = protocol_df["Path"].str.replace("$ROOT/", root_path, regex=False)

# Load SSL model 
ssl = Model(model_name)
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
