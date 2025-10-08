#!/usr/bin/env python
"""script to create protocol for CodecFake database
No additional protocol used, we simply walk through the directory,
and no real audios in this database

/path/to/your/CodecFake/
├── processed/
│   ├── xx.wav
│   ├── . . .

CodecFake.csv
"""
import os
import sys
import csv
import glob

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
dataset_name = 'CodecFake'
data_folder = os.path.join(root_folder, dataset_name, 'processed')
ID_PREFIX = 'CodecF-'
output_csv = dataset_name + '.csv'

# Function to collect metadata from the directory structure
def collect_metadata(data_folder):
    metadata = []
    # List all wav files
    for file_path in sorted(
        glob.glob(os.path.join(data_folder, "**", "*.wav"), recursive=True)
    ):
        relative_path = file_path.replace(root_folder, "$ROOT/")
        # Extract relevant folder names
        parts = os.path.normpath(relative_path).split(os.sep)
        # ['$ROOT', 'CodecFake', 'processed', 'encodec_24khz+p272_103.wav']
        filename = parts[-1]
        info = filename.split('+')
        attack = info[0]
        speaker = info[1].split('_')[0]
        label = 'fake'
        proportion = '-'
        file_id = os.path.splitext(parts[-1])[0]
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
                "Language": 'EN',
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
