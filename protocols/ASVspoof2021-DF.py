"""script to create protocol for ASVspoof2021-DF database
ASVspoof2021-DF.txt:
LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -    TEF2 DF_E_2000013 low_m4a vcc2020 Task1-team20 spoof notrim eval neural_vocoder_nonautoregressive Task1 team20 FF E
TGF1 DF_E_2000024 mp3m4a vcc2020 Task2-team12 spoof notrim eval traditional_vocoder Task2 team12 FF G

/path/to/your/ASVspoof2021-DF/ASVspoof2021_DF_eval/
├── flac/
│   ├── LA_E_xx.flac
│   ├── . . . 
We use the keys downloaded from here https://www.asvspoof.org/index2021.html

ASVspoof2021-DF.csv:
"""
import os
import sys 
import csv 

try:
    import pandas as pd
    from pandarallel import pandarallel
    import torchaudio
except ImportError:
    print("Please install pandas, pandarallel and torchaudio")
    sys.exit(1)


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# used for pandas pd.parallel_apply() to speed up
pandarallel.initialize()

# Define paths
root_folder = '/path/to/your/'
dataset_name = "ASVspoof2021-DF"
ID_PREFIX = "ASV21DF-"
# data_folder shoule be /path/to/your/ASVspoof2021_LA_eval
data_folder = os.path.join(root_folder, dataset_name, 'ASVspoof2021_DF_eval')
protocol_file = dataset_name + ".txt"
output_csv = dataset_name + ".csv"

# Function to read the protocol file and collect metadata
def read_protocol(protocol_file):
    # define converter
    converter_label = lambda x: 'real' if x == 'bonafide' else 'fake'
    # use panads to load the csv and handle the header
    # use converter to handle file ID and label
    metadata = pd.read_csv(
        protocol_file,
        sep=' ',
        header=None,
        names=['Speaker', 'ID', 'Codec', 'Source', 'Attack', 'Label', 'Trim', \
               'Phrase', 'Vocoder', 'UN1', 'UN2', 'UN3', 'UN4' ],
        converters={'Label': converter_label})

    print(metadata.head())
    return metadata

# Collect metadata
def collect_audio_metadata(metadata):
    def __get_audio_meta(row):
    # each input row already has information
    # row['Label'], row['ID'], row['Speaker'], row['Attack'] 
    # we need to add row['Duration'], row['SampleRate'], row['Path'], row['Proportion']
        # filepath
        file_path = os.path.join(data_folder, "flac", f"{row['ID']}.flac")
        file_id = ID_PREFIX + row["ID"]
        # All 2021-DF data is eval
        subset_id = 'eval'
        label = row["Label"]
        # Check if file exists
        if os.path.exists(file_path):
            # use torchaudio.info to increase the speed of loading
            metainfo = torchaudio.info(file_path)
            # sampling rate
            sample_rate = metainfo.sample_rate
            # number of channels
            num_channels = metainfo.num_channels
            # duration (num_frames -> number of sampling points)
            duration = round(metainfo.num_frames / sample_rate, 2)
            # file path
            filepath = file_path.replace(root_folder, "$ROOT/")
            # encoding format
            encoding = metainfo.encoding
            # bit per sample
            bitpersample = metainfo.bits_per_sample
            if num_channels > 1:
                print(f"Warning: File {file_path} has multiple channels.")
            lang = 'EN'
        else:
            duration = -1
            sample_rate = -1
            filepath = ''
            encoding = ''
            bitpersample = -1
            num_channels = -1
            lang = -1
        row['ID'] = file_id
        row['Duration'] = duration
        row['SampleRate'] = sample_rate
        row['Path'] = filepath
        row['Proportion'] = subset_id
        row['AudioChannel'] = num_channels
        row['AudioEncoding'] = encoding
        row['AudioBitSample'] = bitpersample
        row['Language'] = lang
        return row
    metadata = metadata.parallel_apply(lambda x: __get_audio_meta(x), axis=1)
    # Filter out None values (skipped rows)
    return metadata

# Write to CSV
def write_csv(metadata):
    header = ["ID", "Label", "Duration", "SampleRate", "Path", "Attack", "Speaker",\
              "Proportion", "AudioChannel", "AudioEncoding", "AudioBitSample",\
              "Language"]
    metadata = pd.DataFrame(metadata)
    metadata = metadata[header]
    metadata = metadata[metadata['Duration'] != -1].reset_index(drop=True)
    metadata.to_csv(output_csv, index=False)

# Main script
if __name__ == "__main__":
    # Step 1: Read protocol file
    protocol_data = read_protocol(protocol_file)
    # Step 2: Collect metadata
    metadata = collect_audio_metadata(protocol_data)
    # Step 3: Write metadata to CSV
    metadata = metadata.drop(columns=['UN1', 'UN2', 'UN3', 'UN4', 'Codec', 'Source', \
                                      'Trim', 'Phrase', 'Vocoder'], 
                             errors='ignore')
    metadata = metadata.to_dict(orient='records')
    write_csv(metadata)
    print(f"Metadata CSV written to {output_csv}")
