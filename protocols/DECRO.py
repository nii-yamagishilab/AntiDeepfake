#!/usr/bin/env python
"""Script to create a single protocol for DECRO database.

/path/to/your/DECRO/
├── petrichorwq-DECRO-dataset-6fc9884
│   ├── ch_dev/
│   │   ├── xx.wav
│   ├── ch_dev.txt
│   ├── en_dev/
│   │   ├── . . .
│   ├── en_dev.txt
│   ├── . . .

ch_dev.txt:
unknown 20032123 - baidu spoof 
unknown 20032131 - baidu spoof 
unknown 20032139 - baidu spoof 
G0019 T0055G0019S0420 - aidatatang bonafide
G0019 T0055G0019S0202 - aidatatang bonafide
G0019 T0055G0019S0097 - aidatatang bonafide

DECRO.csv:
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
dataset_name = "DECRO"
ID_PREFIX = "DECRO-"
data_folder = os.path.join(root_folder, dataset_name, 'petrichorwq-DECRO-dataset-6fc9884')
output_csv = dataset_name + ".csv"

# Subfolders and their protocol files
subfolders = ["ch_dev", "ch_eval", "ch_train", "en_dev", "en_eval", "en_train"]
protocol_files = {sub: os.path.join(data_folder, f"{sub}.txt") for sub in subfolders}

# Function to check for duplicate IDs
def check_duplicates(metadata):
    duplicates = metadata[metadata.duplicated(subset="ID", keep=False)]
    if not duplicates.empty:
        print("Duplicate IDs found:")
        print(duplicates)
    else:
        print("No duplicate IDs found.")

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    converter_label = lambda x: "real" if x == "bonafide" else "fake"
    metadata = pd.read_csv(
        protocol_file,
        sep=" ",
        header=None,
        names=["Speaker", "ID", "UN", "Attack", "Label"],
        converters={"Label": converter_label},
    )
    return metadata

# Collect metadata
def collect_audio_metadata(metadata, sub_set):
    def __get_audio_meta(row):
        # Determine subset type
        subset_id = (
            "valid" if "dev" in sub_set else ("train" if "train" in sub_set else "test")
        )

        # File path
        file_path = os.path.join(data_folder, sub_set, f"{row['ID']}.wav")

        # Check if file exists
        label = row["Label"]
        # Check if file exists
        if os.path.exists(file_path):
            metainfo = torchaudio.info(file_path)
            sample_rate = metainfo.sample_rate
            num_channels = metainfo.num_channels
            duration = round(metainfo.num_frames / sample_rate, 2)
            relative_path = file_path.replace(root_folder, "$ROOT/")
            parts = os.path.normpath(relative_path).split(os.sep)
            # ['$ROOT', 'DECRO', 'petrichorwq-DECRO-dataset-6fc9884', 'ch_dev', '20032851.wav']
            if 'ch' in parts[3]:
                lang = 'ZH'
            elif 'en' in parts[3]:
                lang = 'EN'
            else:
                lang = '-'
            encoding = metainfo.encoding
            bitpersample = metainfo.bits_per_sample
            if num_channels > 1:
                print(f"Warning: File {file_path} has multiple channels.")
        else:
            print(f"Warning: File {file_path} does not exist, skipping entry.")
            duration = -1
            sample_rate = -1
            relative_path = ""
            encoding = ""
            bitpersample = -1
            num_channels = -1
            lang = -1
        file_id = f'{subset_id}-{lang}-{row["ID"]}'
        row["ID"] = ID_PREFIX + file_id  
        row["Duration"] = duration
        row["SampleRate"] = sample_rate
        row["Path"] = relative_path 
        row["Proportion"] = subset_id
        row["AudioChannel"] = num_channels
        row["AudioEncoding"] = encoding
        row["AudioBitSample"] = bitpersample
        row["Language"] = lang
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
        metadata = metadata.drop(columns=['UN'], errors='ignore')
        # Check for duplicates
        check_duplicates(metadata)
        # Append to combined metadata
        combined_metadata.extend(metadata.to_dict(orient="records"))

    # Write combined metadata to a single CSV
    write_csv(combined_metadata)
    print(f"Combined metadata CSV written to {output_csv}")
