#!/usr/bin/env python
"""script to create protocol for DeepVoice database
Audio segments have to be generated locally using DeepVoice_segmentation.py
We do not use original long audio for testing

/path/to/your/DeepVoice/AUDIO_SEGMENTS/
├── FAKE/
│   ├── xx.wav
│   ├── . . .
├── REAL/
│   ├── xx.wav
│   ├── . . .

DeepVoice.csv:
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
dataset_name = 'DeepVoice'
ID_PREFIX = 'DeepVoice-'
data_folder = os.path.join(root_folder, dataset_name, 'AUDIO_SEGMENTS')
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
        # ['$ROOT', 'DeepVoice', 'AUDIO_SEGMENTS', 'FAKE', 'biden-to-taylor_seg_62.wav']
        label = parts[3].lower()
        attack = 'VC'
        speaker = '-'
        lang = 'EN'
        proportion = '-'
        # ID
        file_id = os.path.splitext(parts[-1])[0]
        try:
            # Load metainfo with torchaudio
            metainfo = torchaudio.info(file_path)
            # Append metadata
            metadata.append({
                # Same file_id can occur in different subfolders
                "ID": ID_PREFIX + attack + "_" + file_id,
                "Label": label,
                "SampleRate": metainfo.sample_rate,
                "Duration": round(metainfo.num_frames / metainfo.sample_rate, 2),
                "Path": relative_path,
                "Attack": attack,
                "Speaker": "-",
                "Proportion": proportion,
                "AudioChannel": metainfo.num_channels,
                "AudioEncoding": metainfo.encoding,
                "AudioBitSample": metainfo.bits_per_sample,
                "Language": lang,
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
