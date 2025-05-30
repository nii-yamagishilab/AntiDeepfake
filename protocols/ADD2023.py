#!/usr/bin/env python
"""script to create protocol for ADD2023 Track1.2 testR2 database
label.txt:
ADD2023_T1.2R2_E_00000003.wav genuine
ADD2023_T1.2R2_E_00000004.wav fake
ADD2023_T1.2R2_E_00000005.wav fake
ADD2023_T1.2R2_E_00000006.wav genuine

/path/to/your/ADD2023/Track1.2/testR2/
├── label.txt
├── log.txt
├── wav/
│   ├── ADD2023_T1.2R2_E_xx.wav
│   ├── . . .

"""

import os
import csv
import torchaudio

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

# Define paths
root_folder = '/path/to/your/'
dataset_name = 'ADD2023'
data_folder = os.path.join(root_folder, dataset_name, 'Track1.2', 'testR2', 'wav')
protocol_file = os.path.join(root_folder, dataset_name, 'Track1.2', 'testR2', 'label.txt')
ID_PREFIX = "ADD2023-"
output_csv = dataset_name + '.csv'

# Function to read the protocol file
def read_protocol(protocol_file):
    # define converter
    converter_label = lambda x: 'real' if x == 'genuine' else 'fake'
    metadata = pd.read_csv(
        protocol_file,
        sep= ' ',
        header= None,
        names=['ID', 'Label'],
        converters={'Label': converter_label})

    print(metadata.head())
    return metadata

# Function to collect additional metadata (duration and sample rate)
def collect_audio_metadata(metadata, root_folder):
    def __get_audio_meta(row):
        file_path = os.path.join(data_folder, f"{row['ID']}")
        label = row["Label"]
        language = '-'
        proportion = 'test'
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
        row["ID"] = ID_PREFIX + row["ID"]
        row["Attack"] = '-'
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
    header = [
        "ID",
        "Label",
        "Duration",
        "SampleRate",
        "Path",
        "Attack",
        "Speaker",
        "Proportion",
        "AudioChannel",
        "AudioEncoding",
        "AudioBitSample",
        "Language",
    ]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata.to_csv(output_csv, index=False)

# Main script
if __name__ == "__main__":
    # Step 1: Read protocol and collect initial metadata
    metadata = read_protocol(protocol_file)
    # Step 2: Collect audio metadata (duration, sample rate, etc.)
    metadata = collect_audio_metadata(metadata, root_folder)
    metadata = metadata.to_dict(orient='records')
    # Step 3: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
