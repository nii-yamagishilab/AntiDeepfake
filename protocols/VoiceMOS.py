#!/usr/bin/env python
"""script to create protocol for VoiceMOS database
No additional protocol used, we simply walk through the directory,

We use this in our training,
which contains only fake audio
/path/to/your/VoiceMOS/
├── main/
│   ├── DATA/
│   │   ├── wav/
│   │   │   ├── xx.wav
│   │   │   ├── . . .
The following code will process the original blizzard challenge data,
which contains same fake audio but additional real audio
/path/to/your/VoiceMOS/
├── main/
│   ├── blizzard_wavs_and_scores_2008_release_version_1/
│   │   ├── A/
│   │   │   ├── /x/x/x/x/xx.wav
│   │   ├── . . .

VoiceMOS.csv:
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
dataset_name = 'VoiceMOS'
data_folder = os.path.join(root_folder, dataset_name, 'main', 'blizzard')
ID_PREFIX = 'VoiceMOS-'
output_csv = dataset_name + '.csv'

# Function to collect metadata from the directory structure
def collect_metadata(data_folder):
    count = 0
    metadata = []
    # List all wav files
    for file_path in sorted(
        glob.glob(os.path.join(data_folder, "**", "*.wav"), recursive=True)
    ):
        relative_path = file_path.replace(root_folder, "$ROOT/")
        # Extract relevant folder names
        parts = os.path.normpath(relative_path).split(os.sep)
# ['$ROOT', 'VoiceMOS', 'main', 'blizzard', 'blizzard_wavs_and_scores_2010_release_version_1', 'N', 'submission_directory', 'mandarin', 'MH1', '2010', 'news', 'wavs', 'news_2010_0095.wav']
        submission = parts[5]
        if submission == 'A':
            label = 'real'
        else:
            label = 'fake'
        attack = '-'
        speaker = '-'
        proportion = '-'
        language = '-'
        # use count to avoid duplicate IDs
        file_id = f'{count}-{os.path.splitext(parts[-1])[0]}'
        count += 1
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
