#!/usr/bin/env python
"""script to create protocol for WaveFake database
No additional protocol used, we simply walk through the directory
There are .wav files under each subfolder, this database contains no real audio

/path/to/your/WaveFake/generated_audio/
├── common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/
│   ├── xx.wav
├── jsut_multi_band_melgan/
│   ├── . . .
├── jsut_parallel_wavegan/
│   ├── . . .
├── . . .

WaveFake.csv:
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
dataset_name = 'WaveFake'
ID_PREFIX = "WF-"
data_folder = os.path.join(root_folder, dataset_name, 'generated_audio')
output_csv = dataset_name + '.csv'

# Function to collect metadata from the directory structure
def collect_metadata(data_folder):
    metadata = []
    # Walk through the directory
    for root, dirs, files in os.walk(data_folder):
        # Prevent os.walk from descending into subdirectories
        if root.count(os.sep) - data_folder.count(os.sep) >= 1:
            dirs[:] = []  # Clear dirs to stop further descent
        for file in files:
            if file.endswith((".wav")):  # Check for audio files
                # File path and ID
                file_path = os.path.join(root, file)
                relative_path = file_path.replace(root_folder, "$ROOT/")
                # Extract relevant folder names
                parts = os.path.normpath(relative_path).split(os.sep)
                # Treat subfolder name as attack ID
                attack = parts[3]
                if 'ljspeech' in attack:
                    lang = 'EN'
                if 'jsut' in attack:
                    lang = 'JA'
                label = 'fake'
                proportion = '-'
                file_id = os.path.splitext(file)[0]
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
