#!/usr/bin/env python
"""script to create protocol for Codecfake database
Please manually mv train/ and dev/ to front

/path/to/your/Codecfake/
├── /train/
│   ├── xx.wav
├── /dev/
│   ├── xx.wav
├── C7/
│   ├── xx.wav
├── label
│   ├── C7.txt
│   ├── dev.txt
│   ├── train.txt

Each protocol .txt looks like this:
SSB13650058.wav real 0
F01_SSB13650058.wav fake 1 
F02_SSB13650058.wav fake 2 

Codecfake.csv:
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
dataset_name = 'Codecfake'
ID_PREFIX = 'Codecf-'
data_folder = os.path.join(root_folder, dataset_name)
output_csv = dataset_name + '.csv'

protocols = ['train', 'dev', 'C7']
protocol_files = {p: os.path.join(data_folder, 'label', f"{p}.txt")\
                  for p in protocols}

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    metadata = pd.read_csv(
        protocol_file,
        sep=" ",
        header=0,
        names=["File", "Label", "F0x"],
    )
    print(metadata.head())
    return metadata

# Collect metadata
def collect_audio_metadata(metadata, sub_set):
    def __get_audio_meta(row):
        # File path
        file_path = os.path.join(data_folder, sub_set, f"{row['File']}")
        file_name = row['File'][:-4]
        file_id = ID_PREFIX + file_name
        if sub_set == 'dev':
            proportion = 'valid'
        elif sub_set == 'C7':
            proportion = 'test'
        elif sub_set == 'train':
            proportion = 'train'
        speaker = '-'
        if row["F0x"] == 0:
            atk = '-'
        elif row["F0x"] == 1:
            atk = 'F01'
        elif row["F0x"] == 2:
            atk = 'F02'
        elif row["F0x"] == 3:
            atk = 'F03'
        elif row["F0x"] == 4:
            atk = 'F04'
        elif row["F0x"] == 5:
            atk = 'F05'
        elif row["F0x"] == 6:
            atk = 'F06'
        elif row["F0x"] == 7:
            atk = 'F07'
        else:
            atk = 'None' 
        label = row['Label']
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
            filepath = ''
            encoding = ''
            bitpersample = -1
            num_channels = -1
            lang = -1
        row["ID"] = file_id
        row["Duration"] = duration
        row["Attack"] = atk
        row["SampleRate"] = sample_rate
        row["Path"] = filepath
        row["Proportion"] = proportion 
        row["AudioChannel"] = num_channels
        row["AudioEncoding"] = encoding
        row["AudioBitSample"] = bitpersample
        row["Language"] = '-'
        row["Speaker"] = speaker
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
        metadata = metadata.drop(columns=['File', 'F0x'], errors='ignore')
        # Append to combined metadata
        combined_metadata.extend(metadata.to_dict(orient="records"))

    # Write combined metadata to a single CSV
    write_csv(combined_metadata)
    print(f"Combined metadata CSV written to {output_csv}")
