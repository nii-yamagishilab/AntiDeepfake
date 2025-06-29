"""This script is used to split the original Deepfake-Eval-2024 audio-data
to train and test partitions based on its metadata protocol.

/path/to/your/Deepfake-Eval-2024/
├── audio-metadata-publish.csv
├── audio-data/
│   ├── xx.mp3
│   ├── xx.wav
│   ├── . . .
├── . . . (data of other modalities are not used)
├── data/ (this script will generate this folder)
│   ├── unsegmented/train/
│   │   ├── real/
│   │   │   ├── (audio files)
│   │   ├── fake/
│   │   │   ├── (audio files)
│   ├── unsegmented/test/
│   │   ├── real/
│   │   │   ├── (audio files)
│   │   ├── fake/
│   │   │   ├── (audio files)
"""
import os 
import shutil
import pandas as pd


__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Define paths
metadata_path = "/path/to/your/Deepfake-Eval-2024/audio-metadata-publish.csv"
audio_source_dir = "/path/to/your/Deepfake-Eval-2024/audio-data"
audio_target_dir = "/path/to/your/Deepfake-Eval-2024/data/unsegmented"

# Read CSV file
metadata = pd.read_csv(metadata_path)

# Create target directories if they don't exist
train_real_dir = os.path.join(audio_target_dir, "train", "real")
train_fake_dir = os.path.join(audio_target_dir, "train", "fake")
test_real_dir = os.path.join(audio_target_dir, "test", "real")
test_fake_dir = os.path.join(audio_target_dir, "test", "fake")

os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_fake_dir, exist_ok=True)
os.makedirs(test_real_dir, exist_ok=True)
os.makedirs(test_fake_dir, exist_ok=True)

# Iterate over metadata and copy files
for _, row in metadata.iterrows():
    filename = row["Filename"]
    set_type = str(row["Finetuning Set"]).strip().lower()
    ground_truth = str(row["Ground Truth"]).strip().lower()
    
    if set_type == "train":
        target_folder = train_real_dir if ground_truth == "real" else train_fake_dir if ground_truth == "fake" else None
    elif set_type == "test":
        target_folder = test_real_dir if ground_truth == "real" else test_fake_dir if ground_truth == "fake" else None
    else:
        continue  # Skip rows that do not belong to Train or Test
    
    if target_folder is None:
        continue  # Skip rows with invalid Ground Truth values
    
    src_path = os.path.join(audio_source_dir, filename)
    dst_path = os.path.join(target_folder, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to {target_folder}")
    else:
        print(f"File not found: {src_path}")
