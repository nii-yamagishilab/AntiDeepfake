#!/usr/bin/env python
"""script to create protocol for VoxCeleb2-Vocoded database
No additional protocol used, we simply walk through the directory,
all files are treated as fake

To generate vocoded data, please follow:
https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing#scrollTo=aNbf2u5odiFv

/path/to/your/VoxCeleb2-Vocoded/
├── hifigan/
|   ├── dev/aac/
│   |   ├── id05541/
│   │   |   ├── 0CI7YNicBFs/
│   │   │   |   ├── xx.wav
│   │   |   ├── . . .
│   │   ├── . . . 
│   ├── . . . 
├── hn_sinc_nsf/
│   ├── . . .
│   │   ├── . . . 
├── . . .

VoxCeleb-Vocoded.csv:
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
dataset_name = 'VoxCeleb2-Vocoded'
data_folder = root_folder + dataset_name
ID_PREFIX = 'VoxCeleb2-Vo-'
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
# ['$ROOT', 'VoxCeleb2-Vocoded', 'hifigan', 'dev', 'aac', 'id04972', 'f9crVjbT1Eg', '00255.wav']
        attack = parts[2]
        trail = parts[6]
        speaker = parts[5]
        label = 'fake'
        proportion = '-'
        language = '-'
        file_id = f"{attack}-{speaker}-{trail}-{os.path.splitext(parts[-1])[0]}"
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
    metadata = collect_metadata(data_folder)
    # Step 2: Write metadata to CSV
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
