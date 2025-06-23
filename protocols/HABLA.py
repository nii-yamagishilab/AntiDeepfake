#!/usr/bin/env python
"""script to create protocol for HABLA: A Dataset of Latin American Spanish Accents for Voice Anti-spoofing

No additional protocol used, we simply walk through the directory

/path/to/your/HABLA/FinalDataset_16khz/
├── CycleGAN/
│   ├── Argentina-Argentina/
│   │   ├── arf_00295-arf_02121/
│   │   │   ├── xx.wav
│   │   ├── . . .
│   ├── Argentina-Chile/
│   │   ├── . . .
├── Diff/
│   ├── . . .
├── Real/
│   ├── . . .
├── . . .

HABLA.csv
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
dataset_name = 'HABLA'
data_folder = os.path.join(root_folder, dataset_name, 'FinalDataset_16khz')
ID_PREFIX = 'HABLA-'
output_csv = dataset_name + '.csv'

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
        # ['$ROOT', 'HABLA', 'FinalDataset_16khz', 'Real', 'Chile', 'clf_07508', 'clf_07508_00799857148.wav']
        # ['$ROOT', 'HABLA', 'FinalDataset_16khz', 'CycleGAN', 'Venezuela-Argentina', 'vem_02484-arm_01523', 'CycleGAN-vem_02484_00993837010-arm_01523_020267.wav']
                attack = parts[3]
                speaker = parts[5]
                proportion = '-'
                if 'Real' in attack:
                    label = 'real'
                else:
                    label = 'fake'
                # ID
                file_id = f"{label}-{os.path.splitext(file)[0]}"
                lang = 'ES'
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
                    "Language": lang,
                })
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
