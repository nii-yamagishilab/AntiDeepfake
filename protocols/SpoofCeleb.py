#!/usr/bin/env python
"""script to create protocol for SpoofCeleb database

/path/to/your/SpoofCeleb/spoofceleb/
├── flac/
│   ├── development/
│   │   ├── a00/ (real audios)
│   │   │   ├── id10310/
│   │   │   │   ├── xx.flac
│   │   │   │   ├── . . . 
│   │   │   ├── id10311/
│   │   │   ├── . . . 
│   │   ├── a06/ (fake audios)
│   │   ├── . . . (fake audios)
│   ├── evaluation/
│   │   ├── . . .
│   ├── train/
│   │   ├── . . .
├── metadata/
│   ├── development.csv
│   ├── evaluation.csv
│   ├── train.csv

development.csv:
file,speaker,attack
a00/id10318/YYsxcZ5saac-00002-006.flac,id10318,a00
a00/id10326/NKOA_Q89wDM-00066-003.flac,id10326,a00
a00/id10349/QGlexcvxGFo-00007-001.flac,id10349,a00

SpoofCeleb.csv:
"""

import os
import sys 
import csv 

try:
    import pandas as pd
    from pandarallel import pandarallel
    import torchaudio
except ImportError:
    print("Please install pandas, pandarallel and torchaudio")
    sys.exit(1)


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# used for pandas pd.parallel_apply() to speed up
pandarallel.initialize()

# Define paths
root_folder = '/path/to/your/'
dataset_name = 'SpoofCeleb'
ID_PREFIX = 'SpoofCeleb-'
data_folder = os.path.join(root_folder, dataset_name, 'spoofceleb')
output_csv = dataset_name + '.csv'

# Subfolders and their protocol files
subfolders = ['development', 'evaluation', 'train']
protocol_files = {sub: os.path.join(data_folder, 'metadata', f"{sub}.csv")\
                  for sub in subfolders}

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    metadata = pd.read_csv(
        protocol_file,
        sep=",",
        header=0,
        names=["File", "Speaker", "Attack"],
    )
    return metadata

# Collect metadata
def collect_audio_metadata(metadata, sub_set):
    def __get_audio_meta(row):
        # File path
        file_path = os.path.join(data_folder, 'flac', sub_set, f"{row['File']}")
        file_name = row['File'][:-5]
        file_id = ID_PREFIX + file_name 
        # All real audios are saved in a00/
        if row['Attack'] == 'a00':
            label = 'real'
        else:
            label = 'fake'
        # Define proportion based on the subset
        if sub_set == 'development':
            proportion = 'valid'
        elif sub_set == 'evaluation':
            proportion = 'test'
        elif sub_set == 'train':
            proportion = 'train'
        # Check if file exists
        if os.path.exists(file_path):
            metainfo = torchaudio.info(file_path)
            sample_rate = metainfo.sample_rate
            num_channels = metainfo.num_channels
            duration = round(metainfo.num_frames / sample_rate, 2)
            filepath = file_path.replace(root_folder, "$ROOT/")
            encoding = metainfo.encoding
            bitpersample = metainfo.bits_per_sample
            if num_channels > 1:
                print(f"Warning: File {file_path} has multiple channels.")
        else:
            print(f"Warning: File {file_path} does not exist, skipping entry.")
            duration = -1
            sample_rate = -1
            filepath = ""
            encoding = ""
            bitpersample = -1
            num_channels = -1
            lang = -1
        row["ID"] = file_id
        row['Label'] = label
        row["Duration"] = duration
        row["SampleRate"] = sample_rate
        row["Path"] = filepath
        row["Proportion"] = proportion 
        row["AudioChannel"] = num_channels
        row["AudioEncoding"] = encoding
        row["AudioBitSample"] = bitpersample
        row["Language"] = '-'
        return row

    metadata = metadata.parallel_apply(lambda x: __get_audio_meta(x), axis=1)
    return metadata

# Write to CSV
def write_csv(metadata):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata = metadata[metadata['Duration'] != -1].reset_index(drop=True)
    metadata.to_csv(output_csv, index=False)

# Main script
if __name__ == "__main__":
    combined_metadata = []

    for sub_set, protocol_file in protocol_files.items():
        print(f"Processing subset: {sub_set}")
        # Check if protocol file exists
        if not os.path.exists(protocol_file):
            print(f"Warning: Protocol file {protocol_file} not found, skipping.")
            continue
        # Read protocol
        protocol_data = read_protocol(protocol_file)
        # Collect metadata
        metadata = collect_audio_metadata(protocol_data, sub_set)
        # Drop unwanted columns
        metadata = metadata.drop(columns=['File'], errors='ignore')
        # Append to combined metadata
        combined_metadata.extend(metadata.to_dict(orient="records"))

    # Write combined metadata to a single CSV
    write_csv(combined_metadata)
    print(f"Combined metadata CSV written to {output_csv}")
