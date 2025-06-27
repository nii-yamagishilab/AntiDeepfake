import os
import random
from pathlib import Path

import torchaudio


__author__ = "Wanying Ge"
__email__ = "gewanying@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Configuration
input_dir = "~/Wav/DeepVoice/AUDIO"
output_dir = "~/Wav/DeepVoice/AUDIO_SEGMENTS"
min_segment_duration = 1.5  # minimum duration for a segment in seconds
max_segment_duration = 15.0  # maximum duration for a segment in seconds
min_overlap_ratio = 0.1  # minimum overlap as a fraction of segment length
max_overlap_ratio = 0.8  # maximum overlap as a fraction of segment length

# Expand user paths
input_dir = os.path.expanduser(input_dir)
output_dir = os.path.expanduser(output_dir)

def create_output_path(input_path, base_output_dir):
    """Create an output path preserving the directory structure."""
    relative_path = os.path.relpath(input_path, start=input_dir)
    return os.path.join(base_output_dir, relative_path)

def split_audio(file_path, output_folder):
    """Split a single audio file into overlapping segments."""
    # Load audio
    y, sr = torchaudio.load(file_path)
    total_duration = y.shape[1] 

    # Generate segments
    segments = []
    start_time = 0
    while 1:
        segment_duration = random.uniform(min_segment_duration, max_segment_duration)
        segment_duration = int(segment_duration*sr)
        overlap_ratio = random.uniform(min_overlap_ratio, max_overlap_ratio)
        end_time = min(start_time + segment_duration, total_duration)
        segments.append((start_time, end_time))
        start_time = end_time - segment_duration * overlap_ratio
        if end_time == total_duration:
            break

    # Save segments
    base_name = Path(file_path).stem
    for i, (start, end) in enumerate(segments):
        segment_file_name = f"{base_name}_seg_{i+1}.wav"
        segment_output_path = os.path.join(output_folder, segment_file_name)

        # Extract and save segment
        start_sample = int(start)
        end_sample = int(end)
        segment_waveform = y[:, start_sample:end_sample]
        print(f"Audio: {base_name}_seg_{i+1}, start at: {start_sample}, end at: {end_sample}")
        torchaudio.save(segment_output_path, segment_waveform, sr)

        print(f"Saved: {segment_output_path}")

def process_directory(input_dir, output_dir):
    """Process all .wav files in the directory tree."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                output_path = create_output_path(root, output_dir)
                # Ensure the output directory exists
                os.makedirs(output_path, exist_ok=True)
                # Split audio and save segments
                split_audio(input_path, output_path)

if __name__ == "__main__":
    process_directory(input_dir, output_dir)
    print("Processing complete!")
