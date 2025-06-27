#!/usr/bin/env python
"""script to create protocol for toy example dataset

/path/to/your_base/toy_example
├── eval
│   ├── LA_E_1066571.wav
│   ├── ...
├── train_dev
│   ├── LA_D_1179848.wav
│   ├── ...
├── protocol.txt
...
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
root_folder = '/gs/bs/tgh-25IAC/ud03523/WORK/misc/Data'
dataset_name = "toy_example"
ID_PREFIX = "ASV19LAdemo-"
# data_folder should be /path/to/your_base/toy_example
data_folder = os.path.join(root_folder, dataset_name)
output_csv = dataset_name
protocol_files = [
    'protocol.txt',
]

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    # define converter
    converter_label = lambda x: 'real' if x == 'bonafide' else 'fake'
    # use panads to load the csv and handle the header
    # use converter to handle file ID and label
    metadata = pd.read_csv(
        protocol_file,
        sep=' ',
        header=None,
        names=['Speaker', 'ID', 'UN', 'Attack', 'Label'],
        converters={'Label': converter_label})

    print(metadata.head())
    return metadata

# Collect metadata
def collect_audio_metadata(metadata):
    def __get_audio_meta(row):
    # each input row already has information
    # row['Label'], row['ID'], row['Speaker'], row['Attack']
    # we need to add row['Duration'], row['SampleRate'], row['Path'], row['Proportion']
        # Determine the subset based on the ID
        if 'T_' in row['ID']:
            subset = 'train_dev'
            subset_id = 'train'
        elif 'D_' in row['ID']:
            subset = 'train_dev'
            subset_id = 'valid'
        else:
            subset = 'eval'
            subset_id = 'test'
        # filepath
        file_path = os.path.join(data_folder, subset, f"{row['ID']}.wav")
        file_id = ID_PREFIX + row["ID"]
        label = row["Label"]

        # Check if file exists
        if os.path.exists(file_path):
            # use torchaudio.info to increase the speed of loading
            metainfo = torchaudio.info(file_path)
            # sampling rate
            sample_rate = metainfo.sample_rate
            # number of channels
            num_channels = metainfo.num_channels
            # duration (num_frames -> number of sampling points)
            duration = round(metainfo.num_frames / sample_rate, 2)
            # file path
            filepath = file_path.replace(root_folder, "$ROOT/")
            # encoding format
            encoding = metainfo.encoding
            # bit per sample
            bitpersample = metainfo.bits_per_sample
            if num_channels > 1:
                print(f"Warning: File {file_path} has multiple channels.")
            lang = 'EN'
        else:
            # If the file doesn't exist, skip the entry
            print(f"Warning: File {file_path} does not exist, skipping entry.")
            duration = -1
            sample_rate = -1
            filepath = ''
            encoding = ''
            bitpersample = -1
            num_channels = -1
        row['ID'] = file_id
        row['Duration'] = duration
        row['SampleRate'] = sample_rate
        row['Path'] = filepath
        row['Proportion'] = subset_id
        row['AudioChannel'] = num_channels
        row['AudioEncoding'] = encoding
        row['AudioBitSample'] = bitpersample
        row['Language'] = lang
        return row
    metadata = metadata.parallel_apply(lambda x: __get_audio_meta(x), axis=1)
    return metadata

# Write to CSV
def write_csv(metadata, subset):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    output_csv_path = output_csv + '_' + subset + '.csv'
    metadata.query('Proportion == "{:s}"'.format(subset)).to_csv(output_csv_path, index=False)

# Main script
if __name__ == "__main__":
    combined_metadata = []
    for protocol_file in protocol_files:
        protocol_file = os.path.join(data_folder,
                                     protocol_file)
        print(f"Processing protocol: {protocol_file}")

        if not os.path.exists(protocol_file):
            print(f"Warning: Protocol file {protocol_file} not found, skipping.")
            continue

        # Step 1: Read sub-protocol
        protocol_data = read_protocol(protocol_file)
        # Step 2: Collect metadata
        metadata = collect_audio_metadata(protocol_data)
        # Step 3: Write metadata to CSV
        # Drop unwanted columns
        metadata = metadata.drop(columns=['UN'], errors='ignore')
        # Append to combined metadata
        combined_metadata.extend(metadata.to_dict(orient="records"))
        
    for subset in ['train', 'valid', 'test']:
        # write train dev eval    
        write_csv(combined_metadata, subset)
        print(f"Metadata CSV written to {output_csv}_{subset}.csv")
