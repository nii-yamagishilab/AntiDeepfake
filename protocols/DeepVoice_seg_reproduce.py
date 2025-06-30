"""This script is used for reproducing the segmented DeepVoice dataset used in our
experiment using the DeepVoice_seg_log.txt provided in this repo.

/path/to/your/DeepVoice/
├── deep-voice-deepfake-voice-recognition.zip (unzip this file)
├── DEMONSTRATION (not used)
├── KAGGLE/
│   ├── AUDIO/
│   │   ├── FAKE/
│   │   │   ├── xx.wav
│   │   │   ├── . . .
│   │   ├── REAL/
│   │   │   ├── xx.wav
│   │   │   ├── . . .
(this script will generate the following folder)
│   ├── AUDIO_SEGMENTS/ 
│   │   ├── FAKE/
│   │   │   ├── xx_seg_x.wav
│   │   │   ├── . . .
│   │   ├── REAL/
│   │   │   ├── xx_seg_x.wav
│   │   │   ├── . . .

DeepVoice_seg_log.txt:
Audio: Obama-to-Biden_seg_1, start at: 0, end at: 209527
Saved: /base_path/Data/DeepVoice/AUDIO_SEGMENTS/FAKE/Obama-to-Biden_seg_1.wav
Audio: Obama-to-Biden_seg_2, start at: 59159, end at: 311206
Saved: /base_path/Data/DeepVoice/AUDIO_SEGMENTS/FAKE/Obama-to-Biden_seg_2.wav
. . .
Audio: musk-original_seg_99, start at: 21350231, end at: 21859457
Saved: /base_path/Data/DeepVoice/AUDIO_SEGMENTS/REAL/musk-original_seg_99.wav
Audio: musk-original_seg_100, start at: 21740914, end at: 22128495
Saved: /base_path/Data/DeepVoice/AUDIO_SEGMENTS/REAL/musk-original_seg_100.wav
"""
import os
import re
from pathlib import Path

import torchaudio


__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

log_file = "DeepVoice_seg_log.txt"
original_audio_dir = os.path.expanduser("~/Wav/DeepVoice/KAGGLE/AUDIO/")
output_base_dir = os.path.expanduser("~/Wav/DeepVoice/KAGGLE/AUDIO_SEGMENTS/")

# Regex pattern to parse log lines
pattern = re.compile(r"Audio:\s*(.*)_seg_(\d+), start at: (\d+), end at: (\d+)")

# Data structure to collect segment info
segments = {}  # filename -> list of (segment_idx, start, end)

# Parse the log
with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            base_name = match.group(1)
            seg_idx = int(match.group(2))
            start = int(match.group(3))
            end = int(match.group(4))
            if base_name not in segments:
                segments[base_name] = []
            segments[base_name].append((seg_idx, start, end))

# Perform re-segmentation
for base_name, seg_list in segments.items():
    # Determine source folder (FAKE or REAL)
    folder = "FAKE" if "-to-" in base_name else "REAL"
    audio_path = os.path.join(original_audio_dir, folder, f"{base_name}.wav")

    if not os.path.exists(audio_path):
        print(f"[WARN] Missing file: {audio_path}")
        continue

    # Load full audio
    waveform, sr = torchaudio.load(audio_path)
    total_samples = waveform.shape[1]

    for seg_idx, start, end in seg_list:
        start = max(0, start)
        end = min(end, total_samples)
        segment = waveform[:, start:end]
        segment_name = f"{base_name}_seg_{seg_idx}.wav"
        output_dir = os.path.join(output_base_dir, folder)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, segment_name)

        torchaudio.save(output_path, segment, sr)
        print(f"[SAVED] {output_path} (samples: {start}-{end})")

print("Re-segmentation completed.")
