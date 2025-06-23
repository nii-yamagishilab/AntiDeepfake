#!/usr/bin/env python
"""script to create protocol for ASVspoof5 database

/path/to/your/ASVspoof5/
├── ASVspoof5.dev.track_1.tsv
├── ASVspoof5.eval.track_1.tsv
├── ASVspoof5.train.tsv
├── ASVspoof5_dev/
│   ├── flac_D/
│   │   ├── xx.flac
│   │   ├── . . .
├── ASVspoof5_eval/
│   ├── flac_E_eval/
│   │   ├── . . . 
├── ASVspoof5_train/
│   ├── flac_T/
│   │   ├── . . . 

ASVspoof5.dev.track_1.tsv:
D_4461 D_0000000169 F - - - AC1 A14 spoof -
D_0461 D_0000000190 M - - - - bonafide bonafide -
D_4579 D_0000000211 M - - - AC1 A09 spoof -
D_1045 D_0000000232 F - - - AC2 A10 spoof -
D_1911 D_0000000253 F - - - - bonafide bonafide -
D_0802 D_0000000274 F - - - AC3 A16 spoof -
D_0913 D_0000000295 F - - - AC2 A11 spoof -

ASVspoof5.csv:
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
dataset_name = 'ASVspoof5'
ID_PREFIX = 'ASV5-'
data_folder = os.path.join(root_folder, dataset_name)
output_csv = dataset_name + '.csv'

protocol_files = [
    'ASVspoof5.dev.track_1.tsv',
    'ASVspoof5.eval.track_1.tsv',
    'ASVspoof5.train.tsv',
]

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    metadata = pd.read_csv(
        protocol_file,
        sep=" ",
        header=0,
        names=["Speaker", "File", "Gender",\
               "UN1", "UN2", "UN3", "UN4",\
               "Attack", "Label", "UN5"],
    )
    return metadata

# Collect metadata
def collect_audio_metadata(metadata):
    def __get_audio_meta(row):
        file_name = row["File"]
        if "T" in file_name:
            proportion = 'train'
            sub_folder = 'ASVspoof5_train'
            file_folder = 'flac_T'
        elif "D" in file_name:
            proportion = 'valid'
            sub_folder = 'ASVspoof5_dev'
            file_folder = 'flac_D'
        elif "E" in file_name:
            proportion = 'test'
            sub_folder = 'ASVspoof5_eval'
            file_folder = 'flac_E_eval'
        file_path = os.path.join(data_folder, sub_folder, file_folder, file_name+'.flac')
        file_id = ID_PREFIX + file_name
        if row['Label'] == 'bonafide':
            label = 'real'
        elif row['Label'] == 'spoof':
            label = 'fake'
        language = 'EN'
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
            language = -1
        row["ID"] = f'{ID_PREFIX}{row["Attack"]}-{file_name}'
        row['Label'] = label
        row["Duration"] = duration
        row["SampleRate"] = sample_rate
        row["Path"] = filepath
        row["Proportion"] = proportion
        row["AudioChannel"] = num_channels
        row["AudioEncoding"] = encoding
        row["AudioBitSample"] = bitpersample
        row["Language"] = language
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
    for protocol_file in protocol_files:
        protocol_file = os.path.join(data_folder, protocol_file)
        print(f"Processing protocol: {protocol_file}")

        # Check if protocol file exists
        if not os.path.exists(protocol_file):
            print(f"Warning: Protocol file {protocol_file} not found, skipping.")
            continue

        # Read protocol
        protocol_data = read_protocol(protocol_file)

        # Collect metadata
        metadata = collect_audio_metadata(protocol_data)

        # Drop unwanted columns
        metadata = metadata.drop(columns=['File', 'UN1', 'UN2', 'UN3',\
                                          'UN4', 'UN5', 'Gender', ],
                                 errors='ignore')
        # Append to combined metadata
        combined_metadata.extend(metadata.to_dict(orient="records"))

    # Write combined metadata to a single CSV
    write_csv(combined_metadata)
    print(f"Combined metadata CSV written to {output_csv}")
