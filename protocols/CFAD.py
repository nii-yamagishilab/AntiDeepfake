#!/usr/bin/env python
"""script to create protocol for CFAD database
No additional protocol used, we simply walk through the directory

/path/to/your/CFAD/CFAD/
├── clean_version/
│   ├── dev_clean/
│   │   ├── fake_clean/
│   │   │   ├── gl/
│   │   │   │   ├── xx.wav
│   │   │   │   ├── . . . 
│   │   │   ├── hifigan/
│   │   │   ├── . . . 
│   │   ├── real_clean/
│   │   │   ├── aishell1/
│   │   │   │   ├── xx.wav
│   │   │   │   ├── . . . 
│   │   │   ├── . . . 
│   ├── test_seen_clean/
│   │   ├── . . .
│   ├── test_unseen_clean/
│   │   ├── . . .
│   ├── train_clean/
│   │   ├── . . .
├── codec_version/
│   ├── dev_codec/
│   ├── . . . 
├── noisy_version/
│   ├── dev_noise/
│   ├── . . . 

CFAD.csv:
"""
import os
import sys
import csv

try:
    import pandas as pd
    import torchaudio
except ImportError:
    print("Please install pandas and torchaudio")
    sys.exit(1)


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Define paths 
root_folder = '/path/to/your/'
dataset_name = 'CFAD'
# data_folder should be /path/to/your/CFAD
data_folder = os.path.join(root_folder, dataset_name, 'CFAD')
ID_PREFIX = "CFAD-"
output_csv = dataset_name + ".csv"

# Function to collect metadata from the directory structure
def collect_metadata(data_folder):
    metadata = []
    # Walk through the directory
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                # File path and ID
                file_path = os.path.join(root, file)
                relative_path = file_path.replace(root_folder, "$ROOT/")
                # Extract relevant folder names
                parts = os.path.normpath(relative_path).split(os.sep)
# ['$ROOT', 'CFAD', 'CFAD', 'codec_version', 'test_seen_codec', 'fake_codec', 'lpcnet', 'SSB09660003_lpcnet_m4a.wav']
                version = parts[3]
                # Define proportion based on 'test_seen_codec' part
                if 'test' in parts[4]:
                    proportion = 'test'
                elif 'dev' in parts[4]:
                    proportion = 'valid'
                elif 'train' in parts[4]:
                    proportion = 'train'
                # Define label based on 'fake_codec' part
                if 'fake' in parts[5]:
                    label = 'fake'
                if 'real' in parts[5]:
                    label = 'real'
                # Define attack based on 'lpcnet' part
                # for real audio, this will be the source database name
                attack = parts[6]
                # Define speaker ID based on database naming style:
                # '<SSB09660003>_lpcnet_m4a.wav'
                speaker = parts[7].split("_")[0]
                # ID
                file_id = f"{version}-{proportion}-{label}-{os.path.splitext(file)[0]}"
                if os.path.exists(file_path):
                    try:
                        # Load metainfo with torchaudio
                        metainfo = torchaudio.info(file_path)
                        # Append metadata
                        metadata.append({
                            "ID": ID_PREFIX + file_id,
                            "Label": label,
                            "SampleRate": metainfo.sample_rate,
                            "Duration": round(metainfo.num_frames / metainfo.sample_rate, 2), 
                            "Path": relative_path,
                            "Attack": attack,
                            "Speaker": speaker, 
                            "Proportion": proportion,
                            "AudioChannel": metainfo.num_channels,
                            "AudioEncoding": metainfo.encoding,
                            "AudioBitSample": metainfo.bits_per_sample,
                            "Language": 'ZH',
                        })
                    except Exception as e:
                    # Handle any exception and skip this file
                        print(f"Error: Could not load file {file_path}. Skipping. Reason: {e}")
    return metadata
                
# Write metadata to CSV
def write_csv(metadata):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata.to_csv(output_csv, index=False)

# Main script
if __name__ == "__main__":
    # Step 1: Collect metadata
    metadata = collect_metadata(data_folder)
    # Step 2: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
