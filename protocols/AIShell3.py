#!/usr/bin/env python
"""script to create protocol for AIShell3 databse
No additional protocol used, we simply walk through the directory,
and no fake audios in this database

/path/to/your/AIShell3/
├── test/
│   ├── wav/
│   │   ├── SSB0005/
│   │   │   ├── xx.wav
│   │   ├── . . . 
├── train/
│   ├── . . . 

AIShell3.csv:
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
dataset_name = 'AIShell3'
data_folder = os.path.join(root_folder, dataset_name)
ID_PREFIX = 'AIShell3-'
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
# ['$ROOT', 'AIShell3', 'train', 'wav', 'SSB1567', 'SSB15670385.wav']
                speaker = parts[4]
                subset = parts[2]
                if 'train' in subset:
                    proportion = 'train'
                elif 'test' in subset:
                    proportion = 'test'
                attack = '-'
                label = 'real'
                language = 'ZH'
                file_id = f"{speaker}-{os.path.splitext(file)[0]}"
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
                        "Language": language,
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
