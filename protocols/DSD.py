#!/usr/bin/env python
"""script to create protocol for DSD corpus
We need to download the official protocol to get the ground-truth label, link:
https://zenodo.org/records/13788455

DSD_corpus_v1.csv:
Utterence name (file name),TTS or VC,Is multi-speaker?,Language,Noise type 1,Source link,utt,group,Speaker name,Gender,Age,label,Model,subset,duration
09MKIS0040_12815.wav,-,No,Korean,-,-,09MKIS0040_12815,AIHUB,09MKIS0040,Male,Adult,bonafide,-,train,6.25
06FKMJ0055_08352.wav,-,No,Korean,-,-,06FKMJ0055_08352,AIHUB,06FKMJ0055,Female,Adult,bonafide,-,train,5.5
12MWKH0048_000026.wav,TTS,No,Korean,-,-,12MWKH0048_000026,VITS-AIHUB,12MWKH0048,Male,Adult,spoof,VITS,eval,1.552
12MWKH0048_000014.wav,TTS,No,Korean,-,-,12MWKH0048_000014,VITS-AIHUB,12MWKH0048,Male,Adult,spoof,VITS,eval,1.6

/path/to/your/DSD/
├── xx.wav
├── . . .

DSD.csv:
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
dataset_name = 'DSD'
data_folder = os.path.join(root_folder, dataset_name)
# We include the protocol in our code
protocol_file = 'DSD_corpus_v1.csv' 
ID_PREFIX = 'DSD-'
output_csv = dataset_name + '.csv' 

# Function to collect metadata from the directory structure
def collect_metadata(protocol_file):
    # Open and read the CSV file
    with open(protocol_file, mode='r', encoding='utf-8') as file:
        metadata = []
        reader = csv.DictReader(file)
        # Skip the first line by advancing the iterator
        next(reader, None)
        # Read each line for meta data collection
        for row in reader:
            utterance_name = row['Utterence name (file name)']
            label = row['label']
            if label == 'bonafide':
                label = 'real'
            if label == 'spoof':
                label = 'fake'
            speaker = row['Speaker name']
            attack = row['Model']
            language = row['Language']
            if language == 'Korean':
                language = 'KO'
            elif language == 'English':
                language = 'EN'
            elif language == 'Japanese':
                language = 'JA'
            proportion = row['subset']
            if proportion == 'eval':
                proportion = 'test'
            if proportion == 'dev':
                proportion = 'valid'
            file_path = os.path.join(data_folder, utterance_name)
            relative_path = file_path.replace(root_folder, "$ROOT/")
            file_id = os.path.splitext(utterance_name)[0]
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
    metadata = collect_metadata(protocol_file)
    # Step 2: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
