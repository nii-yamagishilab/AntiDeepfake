"""This script segments the original Deepfake-Eval-2024 audio files into shorter chunks.

Each audio file is split into multiple segments of <duration> seconds.  
If the final segment is shorter than <duration>, it will be saved at its original length.

The resulting segments will be stored in a subfolder named <duration>s (e.g., 4s, 10s,..)

Note: This script assumes you have already run `DeepfakeEval2024_preprocess.py`.

/path/to/your/Deepfake-Eval-2024/
├── data/
│   ├── unsegmented/ 
│   │   ├── train/
│   │   │   ├── fake/
│   │   │   │   ├── xx.wav/mp3/m4a
│   │   │   ├── real/
│   │   │   │   ├── xx.wav/mp3/m4a
│   │   ├── test/
│   │   │   ├── fake/
│   │   │   │   ├── xx.wav/mp3/m4a
│   │   │   ├── real/
│   │   │   │   ├── xx.wav/mp3/m4a

│   ├── segmented/
│   │   ├── train/
│   │   │   ├── 4s/
│   │   │   │   ├── fake/
│   │   │   │   │   ├── xx_seg_x.wav/mp3/m4a
│   │   │   │   ├── real/
│   │   │   │   │   ├── xx_seg_x.wav/mp3/m4a
│   │   │   ├── 10s/
│   │   │   ├── . . .
│   │   ├── test/
│   │   │   ├── 4s/ . . .
│   │   │   ├── 10s/ . . .
│   │   │   ├── . . .
"""
import os
import math
from pathlib import Path

import librosa
import soundfile as sf


__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Set your base paths
input_base = Path("/path/to/your/Deepfake-Eval-2024/data/unsegmented")
output_base = Path("/path/to/your/Deepfake-Eval-2024/data/segmented")

# Durations in seconds
durations = [4, 10, 13, 30, 50]

# File extensions to process
audio_exts = [".mp3", ".m4a", ".wav", ".flac"]

def segment_audio(audio_path, output_dir, duration_sec):
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Use original sampling rate
        total_samples = len(y)
        samples_per_segment = int(duration_sec * sr)
        num_segments = math.ceil(total_samples / samples_per_segment)

        for i in range(num_segments):
            start = i * samples_per_segment
            end = min((i + 1) * samples_per_segment, total_samples)
            segment = y[start:end]
            segment_filename = f"{audio_path.stem}_seg{i+1}{audio_path.suffix}"
            sf.write(output_dir / segment_filename, segment, sr)
    except Exception as e:
        print(f"Failed to process {audio_path}: {e}")

def process_directory(split):
    for label in ['real', 'fake']:
        input_dir = input_base / split / label
        for audio_file in input_dir.rglob("*"):
            if audio_file.suffix.lower() in audio_exts:
                for d in durations:
                    output_dir = output_base / split / f"{d}s" / label
                    output_dir.mkdir(parents=True, exist_ok=True)
                    segment_audio(audio_file, output_dir, d)

# Run processing for both train and test
for split in ['train', 'test']:
    process_directory(split)
