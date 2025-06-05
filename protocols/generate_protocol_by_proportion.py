"""Script to merge individual protocol CSV files into
train.csv, valid.csv, and test.csv files.

- Protocols listed in `train_files` are merged.
  Audio files labeled with Proportion=='valid' are saved to valid.csv.
  All other files are saved to train.csv.

- Protocols listed in test_files are merged into test.csv.
"""

import os
import pandas as pd


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, NII Yamagishi Lab"

def check_and_remove_duplicates(file_path, target_column='ID'):
    # Read the CSV file
    df = pd.read_csv(file_path)
    duplicates = df[df.duplicated(subset=[target_column], keep=False)] 
    if not duplicates.empty:
        print(f"Duplicates found in {file_path}:\n{duplicates}\n")
        # Remove duplicate IDs within the file
        df = df.drop_duplicates(subset=[target_column], keep=False)
        print("Duplicates removed\n")
    else:
        print(f"{file_path} contains no duplicates")
    return df

def generate_datasets_with_valid_split(train_files, test_files, output_folder="."):
    def process_files(file_list, valid_split=False):
        """ Concadating different protocols, 
            and preserving some of the files for validation
        """
        combined_df = pd.DataFrame()  # Initialize an empty DataFrame
        combined_valid = pd.DataFrame()
        for file in file_list:
            file_path = file 
            if not os.path.exists(file_path):
                print(f"Warning: File {file} not found, skipping...")
                continue
            # Check if there are duplicate IDs in the protocol,
            # SpeechBrain dataloader will stop initialization if there is any
            df = check_and_remove_duplicates(file_path) 
            # If we're processing validation data, split based on 'Proportion' column
            if valid_split and 'Proportion' in df.columns:
                valid_df = df[df['Proportion'] == 'valid']
                df = df[df['Proportion'] != 'valid']
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                combined_valid = pd.concat([combined_valid, valid_df], ignore_index=True)
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        if valid_split:
        # When need to split the dataset to train and validation
            return combined_df, combined_valid
        else:
        # When use the whole dataset
            return combined_df
    
    def calculate_duration(df):
        """
        Calculate the total duration in hours from the 'Duration' column (in seconds).
        """
        if 'Duration' in df.columns:
            total_duration_sec = df['Duration'].sum()  # Sum of all durations in seconds
            total_duration_hr = total_duration_sec / 3600  # Convert to hours
            return total_duration_hr
        else:
            print("Warning: 'Duration' column not found.")
            return 0
    
    # Process train files
    train_combined, valid_combined = process_files(train_files, valid_split=True)
    # Shuffle
    train_combined = train_combined.sample(frac=1, random_state=42)
    # Write files
    print(f"Writing final protocols . . .")
    train_combined.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    print(f"train.csv saved with {len(train_combined)} rows.")
    valid_combined.to_csv(os.path.join(output_folder, "valid.csv"), index=False)
    print(f"valid.csv saved with {len(valid_combined)} rows.")
    # Calculate and print the total duration in hours for train and valid
    print(f"Total duration for train set: {calculate_duration(train_combined):.2f} hours")
    print(f"Total duration for valid set: {calculate_duration(valid_combined):.2f} hours")
    
    # Process test files
    test_combined = process_files(test_files)
    if not test_combined.empty:
        test_combined.to_csv(os.path.join(output_folder, "test.csv"), index=False)
        print(f"test.csv saved with {len(test_combined)} rows.")
        # Calculate and print the total duration in hours for test
        print(f"Total duration for test set: {calculate_duration(test_combined):.2f} hours")
    else:
        print("No data to save for test set.")

train_files = [
    "AIShell3.csv",            
    "ASVspoof2019-LA.csv",     
    "ASVspoof2021-LA.csv",
    "ASVspoof2021-DF.csv",
    "ASVspoof5.csv",
    "CFAD.csv",
    "CNCeleb2.csv",            
    "Codecfake.csv",
    "CodecFake.csv",
    "CVoiceFake_Large.csv",    
    "DECRO.csv",
    "DFADD.csv",
    "DiffusionDeepfake.csv",   
    "DiffSSD.csv",
    "DSD.csv",
    "FLEURS.csv",
    "FLEURS-R.csv",
    "HABLA.csv",
    "LibriTTS.csv",
    "LibriTTS-R.csv",
    "LibriTTS-Vocoded.csv",    
    "LJSpeech.csv",     
    "MLAAD_v5.csv",
    "MLS.csv",
    "SpoofCeleb.csv",          
    "VoiceMOS.csv",
    "VoxCeleb2.csv",
    "VoxCeleb2-Vocoded.csv",
    "WaveFake.csv",
]

test_files = [
    "ADD2023.csv",
    "DeepVoice.csv",
    "FakeOrReal.csv",
    "FakeOrReal_norm.csv",
    "In-the-Wild.csv",
]

generate_datasets_with_valid_split(
    train_files,
    test_files,
)
