#!/usr/bin/env python
"""script to create protocol for In-the-wild database
In-the-Wild.txt (.csv)/
file,speaker,label
0.wav,Alec Guinness,spoof
1.wav,Alec Guinness,spoof

path/to/your/In-the-Wild/release_in_the_wild/
├── xx.wav
├── . . . 
├── meta.csv (the protocol .txt(.csv) used in this script)

In-the-Wild.csv:
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
__copyright__ = "Copyright 2025, NII Yamagishi Lab"

# Define paths 
root_folder = '/path/to/your/'
dataset_name = "In-the-Wild"
ID_PREFIX = "ItW-"
# data_folder should be /path/to/your/In-the-Wild
data_folder = os.path.join(root_folder, dataset_name, 'release_in_the_wild')
protocol_file = dataset_name + ".txt"
output_csv = dataset_name + ".csv"

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    # define converter 
    # to rename .wav from ID
    converter_name = lambda x: os.path.splitext(x)[0]
    # to change bona-fide -> real, spoof -> fake
    converter_label = lambda x: 'real' if x == 'bona-fide' else 'fake'

    # use panads to load the csv and handle the header
    # use converter to handle file ID and label
    metadata = pd.read_csv(
        protocol_file, 
        sep =',', 
        header=0,
        converters = {'file': converter_name, 'label': converter_label})

    # rename the columns to ID, Label, Speaker
    metadata = metadata.rename(
        columns = {'file': "ID", 'label': 'Label', 'speaker': 'Speaker'})

    # add missing column of attacks
    metadata['Attack'] = '-'

    metadata['Proportion'] = '-'

    # print a few lines for sanity check
    print(metadata.head())

    return metadata

# Function to collect additional metadata (duration and sample rate)
def collect_audio_metadata(metadata, data_folder):

    def __get_audio_meta(row):
        # each input row has information
        # row['Label'], row['ID'], row['Speaker'], row['Attack']
        # we need to add row['Duration'], row['SampleRate'], row['path']

        # filepath
        file_path = os.path.join(data_folder, f"{row['ID']}.wav")
        fileid = ID_PREFIX + row["ID"]
        
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

        else:
            # If the file doesn't exist, skip the entry
            print(f"Warning: File {file_path} does not exist, skipping entry.")
            duration = -1
            sample_rate = -1
            filepath = ''
            encoding = ''
            bitpersample = -1
            num_channels = -1

        row['ID'] = fileid
        row['Duration'] = duration
        row['SampleRate'] = sample_rate
        row['Path'] = filepath
        row['AudioChannel'] = num_channels
        row['AudioEncoding'] = encoding
        row['AudioBitPerSample'] = bitpersample
        row['Language'] = 'EN'
        return row

    metadata = metadata.progress_apply(lambda x: __get_audio_meta(x), axis=1)
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
    # Step 1: Read protocol and collect initial metadata
    metadata = read_protocol(protocol_file)
    # Step 2: Collect audio metadata (duration, sample rate, etc.)
    metadata = collect_audio_metadata(metadata, data_folder)
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
