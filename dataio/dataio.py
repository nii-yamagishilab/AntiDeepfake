"""This script defines dataio_prepare() used to load and preprocess audio for
train, valid, and test datasets.
"""
import random

import torch
import torchaudio
import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.sampler import DynamicBatchSampler

from dataio.rawboost import process_Rawboost_feature


__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

def filter_by_duration(dataset, min_dur=None, max_dur=None):
    """Filter dataset by audio length.
    """
    # Convert [Duration] column from str to float
    for _, value in dataset.data.items():
        value["Duration"] = float(value["Duration"])
    # Set a threshold for filtering
    filtered_dataset = dataset.filtered_sorted(
        sort_key="Duration",
        key_min_value={"Duration": min_dur} if min_dur is not None else {},
        key_max_value={"Duration": max_dur} if max_dur is not None else {},
    )
    return filtered_dataset


def dataio_prepare(hparams):
    """Prepare train and valid dataset.
    """
    data_folder = hparams["data_folder"]
    # Load datasets and replace the placeholder '$ROOT' to the actual data folder path
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"ROOT": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"ROOT": data_folder},
    )
    # Filter dataset by audio length, we only filter out very short audio by default.
    # Model forward pass will break if input audio is shorter than a few frames,
    # this is likely to happen during inference when padding is disabled.
    train_data = filter_by_duration(
        train_data, min_dur=hparams["min_duration"], max_dur=hparams["max_duration"]
    )
    valid_data = filter_by_duration(
        valid_data, min_dur=hparams["min_duration"], max_dur=hparams["max_duration"]
    )

    # === Dataloader behaviour for TRAIN and VALID data ===
    # Step[1]: Define which column should be readed from the .csv protocol
    @sb.utils.data_pipeline.takes("Path", "Label", "SampleRate", "AudioChannel")
    # Step[2]: Define which value should be returned by data loading pipeline
    # If you want to do extra processing and return something new,
    # you need to 1st: add a new column to @data_pipeline.takes (for exalmple, "Duration"),
    # and 2nd: do some new processing and calculation, and 
    # add the new return value to @data_pipeline.provides (for example, wav_len_in_50kHz)
    # This, however, only controls the behavior for single audio loading,
    # to return this new value batch-wisely, you also need to 
    # 3rd: add wav_len_in_50kHz to sb.dataio.dataset.set_output_keys defined in below
    # As it regulates a same behavior among all chosen datasets
    @sb.utils.data_pipeline.provides("wav", "logit")
    # The following function behaves like __getitem__(): reads [1] and returns [2]
    def audio_pipeline(Path, Label, SampleRate, AudioChannel):
        # Load the audio file
        wav = sb.dataio.dataio.read_audio(Path)
        # Convert multiple channel to single channel
        if int(AudioChannel) != 1:
            # dim - the dimension to be reduced
            wav = wav.mean(dim=1)
        # Make sure waveform shape is [T, ]
        assert wav.dim() == 1, wav.dim()
        ### Cut very long audios into random but shorter length ###
        original_len = wav.shape[0]
        # Threshold is 13 seconds
        if original_len > 13 * int(SampleRate):
            # Get a random but shorter length between 10s and 13s
            segment_length = random.randint(
                10 * int(SampleRate), 
                13 * int(SampleRate)
            )
            # Find the max start point where a segment_length can be cutted
            # i.e., to cut a 13s segment from a 16.1s audio, max_start will be 3.1s
            max_start = original_len - segment_length
            # Find a random start point between 0 and max_start
            start_idx = random.randint(0, max_start)
            # Get the shorter segment
            wav = wav[start_idx:start_idx + segment_length]
        assert wav.shape[0] <= 13 * int(SampleRate), "Waveform segment is too long."
        ### Finish cutting ###
        # Resampling to 16kHz if not
        if int(SampleRate) != 16000:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=int(SampleRate),
                new_freq=16000,
            )
        # RawBoost augmentation
        if hparams['use_da']:
            wav = process_Rawboost_feature(
                wav,
                sr=16000,
                args=hparams['rawboost'],
                algo=hparams['rawboost']['algo'],
            ) 
        # Normalize waveform 
        with torch.no_grad():
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        yield wav
        # Set the ground-truth training target, 1 for real audio and 0 for fake 
        if Label == "real":
            logit = 1
        elif Label == "fake":
            logit = 0
        else:
            raise Exception(
                f"Unrecognized label: {Label}, should either be real or fake"
            )
        yield logit

    # Desired datasets to use audio_pipeline()
    sb.dataio.dataset.add_dynamic_item([train_data, valid_data], audio_pipeline)
    # Desired dataset to return pre-defined keys batch-wisely
    sb.dataio.dataset.set_output_keys(
        datasets=[train_data, valid_data],
        output_keys=["id", "wav", "logit"]
    ) 

    # Using DynamicBatch Sampler for training, batch size is not fixed during training,
    # and each mini-batch will contain audios with similar length,
    # and total duration of each mini-batch will be limited to hparams["max_batch_length"]
    dynamic_hparams = hparams["dynamic_batch_sampler_train"]
    train_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda x: float(x["Duration"]),
        **dynamic_hparams,
    )
    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        # Pad zeros to shorter audio waveform, to form a fixed length mini-batch
        "collate_fn": PaddedBatch,
        "num_workers": hparams["num_workers"],
        "pin_memory": True,
    }
    valid_loader_kwargs = {
        "batch_size": hparams["valid_dataloader_options"]["batch_size"],
        "collate_fn": PaddedBatch,
        "num_workers": hparams["num_workers"],
        "pin_memory": True,
    }

    return train_data, valid_data, train_loader_kwargs, valid_loader_kwargs

def dataio_prepare_test(hparams):
    """Prepare test dataset.
    """
    data_folder = hparams["data_folder"] 
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"ROOT": data_folder},
    )
    # We only filter out very short audio in testing
    test_data = filter_by_duration(
        test_data, min_dur=hparams["min_duration"], max_dur=None,
    )
    # === Dataloader behaviour for TEST data -- no trimming, no augmentation ===
    @sb.utils.data_pipeline.takes("Path", "Label", "SampleRate", "AudioChannel")
    @sb.utils.data_pipeline.provides("wav", "logit")
    def test_audio_pipeline(Path, Label, SampleRate, AudioChannel):
        wav = sb.dataio.dataio.read_audio(Path)
        if int(AudioChannel) != 1:
            wav = wav.mean(dim=1)
        assert wav.dim() == 1, wav.dim()
        if int(SampleRate) != 16000:
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=int(SampleRate),
                new_freq=16000,
            )
        with torch.no_grad():
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        yield wav
        if Label == "real":
            logit = 1
        elif Label == "fake":
            logit = 0
        else:
            raise Exception(
                f"Unrecognized label: {Label}, should either be real or fake"
            )
        yield logit
    
    sb.dataio.dataset.add_dynamic_item([test_data], test_audio_pipeline)
    sb.dataio.dataset.set_output_keys([test_data], ["id", "wav", "logit"]) 
    # We do not need test_loader_kwargs{}, if it is not specified,
    # speechbrain will build a test dataloader with batch_size=1
    # and without padding automatically
    return test_data
