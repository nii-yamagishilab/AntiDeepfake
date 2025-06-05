"""This script is used for model training. We have included some comments to help
users better understand the code.
"""
import os
import sys
import time
import random
from pathlib import Path

import pandas as pd
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
import torch
import torchaudio
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain import Stage
from speechbrain.core import AMPConfig
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

from utils import load_weights, set_random_seed, compute_eer, process_Rawboost_feature

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

logger = get_logger(__name__)

class SSLBrain(sb.core.Brain):
    def __init__(self, *args, **kargs):
        super(SSLBrain, self).__init__(*args, **kargs)
        
        # in case model is defined via yaml file
        # if case the model is not defined in yaml, add them here
        # to the checkpointer recoverable settings
        # self.checkpointer.add_recoverables({})
        #
        # furthermore, define load_checkpoint to manually load
        # modules if they are added not via yaml file

        # load pre-trained weights if necessary
        # see comment in the function
        if self.hparams.use_pretrained:
            self._init_model()
        
        return

    def _init_model(self):
        """ _init_model()
        Load pre-trained weights

        This is done before loading the checkpoint, after loading the initial
        fairseq model
        """
        # iterate over all modules specified in yaml
        for key in self.hparams.pretrained_weights.keys():
            pretrained_path = Path(self.hparams.pretrained_weights[key])
        
            # find the module that match the name specfied in yaml
            if pretrained_path.exists() and hasattr(self.modules, key):
                logger.info("Loading pre-trained weights {:s} from {:s}".format(
                    key, str(pretrained_path))
                )
                # load pre-trained model for specific module
                load_weights(getattr(self.modules, key).state_dict(), pretrained_path)

        return

    def compute_forward(self, batch, stage):
        """Computes forward pass through SSL model and returns binary class predictions 
        """
        input_data = batch["wav"].data.to(device=self.device, non_blocking=True)        
        preds = self.modules.detector(input_data)
        return preds

    def compute_objectives(self, preds, batch, stage):
        """Compute loss; log prediction and ground-truth for valid EER calculation
        """
        # ground-truth label
        logit = batch["logit"].to(device=self.device, non_blocking=True)
        # Cross-entropy loss of <predictions, ground-truth labels>
        loss = torch.nn.functional.cross_entropy(preds, logit)
        # Store scores and labels for EER calculation
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                ids=batch["id"],
                scores=preds[:, 1],
                labels=logit,
            )
        
        objectives = {
            "loss": loss,
        }
        objectives["backprop_loss"] = loss
        return objectives 

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={}
    ):
        logger = sb.core.logger
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set,
                stage=sb.Stage.TRAIN,
                **train_loader_kwargs,
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )
        self.on_fit_start()
        
        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()
            
        # Iterate epochs
        for epoch in epoch_counter:
            # fit using customized function
            self._fit_train_customized(
                train_set=train_set,
                valid_set=valid_set,
                epoch=epoch,
                enable=enable,
                valid_step=self.hparams.valid_step,
            )

        # original validation API
        self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

    def _fit_train_customized(
        self,
        train_set,
        valid_set,
        epoch,
        enable,
        valid_step=100
    ):
        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()
        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        steps_since_ckpt = 0
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            if self.profiler is not None:
                self.profiler.start()
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                steps_since_ckpt += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss,
                    self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)
                
                if self.profiler is not None:
                    self.profiler.step()
                    if self.profiler.step_num > self.tot_prof_steps:
                        logger.info(
                            "The profiler finished, training is stopped."
                        )
                        self.profiler.stop()
                        quit()
                
                ## customize the code to do validation every N steps
                if self.step % valid_step == 0:
                    self._fit_valid_customized(valid_set, epoch, enable)
                    if self._should_save_intra_epoch_ckpt(
                        last_ckpt_time,
                        steps_since_ckpt
                    ):
                        # Checkpointer class will handle running this on main only
                        self._save_intra_epoch_ckpt()
                        last_ckpt_time = time.time()
                        steps_since_ckpt = 0
                
        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        return

    def _fit_valid_customized(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is None:
            return 0.0

        self.on_stage_start(sb.Stage.VALID, epoch)
        
        self.modules.eval()
        avg_valid_loss = 0.0
        self.step = 0
        with torch.no_grad():
            for batch in valid_set:
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                avg_valid_loss = self.update_average(loss, avg_valid_loss)
            self.step = 0
            self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)
        self.modules.train()
        return 
    
    def fit_batch(self, batch):
        """ compute_forward();compute_objectives();optimizers_step()
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        
        # Managing automatic mixed precision
        with self.no_sync(not should_step):
            preds = self.compute_forward(batch, Stage.TRAIN)
            objectives = self.compute_objectives(
                preds, batch, Stage.TRAIN
            )
            
            self.scaler.scale(
                 objectives["backprop_loss"]/ self.grad_accumulation_factor
            ).backward()
            
            objectives["total_loss"] = objectives["backprop_loss"].detach()

        if should_step:
            self.optimizers_step()
            self.on_fit_batch_end(objectives)
        return objectives["backprop_loss"].detach()

    def on_fit_batch_end(self, objectives):
        """Called after fit_batch(), updates learning rate and does per-step logging.
        """
        self.hparams.lr_scheduler(self.optimizer)
        # Perform step-wise logging
        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):
            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in objectives.items()
            }
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss
            
            if hasattr(self, "valid_epoch_loss"):
                log_dct["valid_epoch_loss"] = self.valid_epoch_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(
                    stats_meta=log_dct,
                )

    def evaluate_batch(self, batch, stage):
        """Return objectives on non-training stages; Log validation loss to logger
        """
        preds = self.compute_forward(batch, stage=stage)
        objectives = self.compute_objectives(preds, batch, stage=stage)
        if stage == sb.Stage.VALID:
            # Accumulate validation loss
            self.valid_epoch_loss = (
                self.valid_epoch_loss + objectives["backprop_loss"].detach().cpu().item()
                if hasattr(self, "valid_epoch_loss")
                else objectives["backprop_loss"].detach().cpu().item()
            )
        return objectives["backprop_loss"].detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            if stage == sb.Stage.VALID:
                self.valid_epoch_loss = 0.0

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of each epoch"""
        stage_stats = {"loss": stage_loss}
        # Only record loss during training
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        if stage == sb.Stage.VALID:
            # Record loss and EER during validation
            # field='DER' returns EER if no threshold is passed,
            # see speechbrain.utils.metric_stats.BinaryMetricStats()
            stage_eer = self.error_metrics.summarize(
                field="DER",
                threshold=None,
            )
            # Logging EER in percentage
            stage_stats = {
                "loss": stage_loss,
                "eer": stage_eer*100,
            }
            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=None,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                # Only save top-1 best models, to save storage space
                num_to_keep=1,
                # Best model selection is based on validation EER
                meta={"EqualErrorRate": stage_stats["eer"]},
                min_keys=["EqualErrorRate"],
            )

    def evaluate(self, dataset, min_key, loader_kwargs={}):
        """Called for final evaluation and saving score .csv file
        """
        loader_kwargs["ckpt_prefix"] = None
        dataset = self.make_dataloader(
            dataset, sb.Stage.TEST, **loader_kwargs
        )
        # Load the best model based on the given key
        self.on_evaluate_start(min_key=min_key)

        self.modules.eval()

        score_preds = sb.utils.metric_stats.ClassificationStats()

        with torch.no_grad():
            for batch in dataset:
                preds = self.compute_forward(batch, stage=sb.Stage.TEST)
                score_preds.append(
                    ids=batch["id"],
                    predictions=preds,
                    targets=batch["logit"],
                )
            score_data = {
                "ID": score_preds.ids,
                "Score": [score.tolist() for score in score_preds.predictions],
                "Label": [label.item() for label in score_preds.targets],
            }
            df_score = pd.DataFrame(score_data)
            # Write the final score
            df_score.to_csv(self.hparams.score_path, index=False)
        print("Scores saved to {}".format(self.hparams.score_path))

def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]            
    # Load datasets and replace the placeholder '$ROOT' to the actual dataset path
    # Data loading will work fine even if '$ROOT' is not written in df['Path'], i.e.,
    # when you choose to write full audio path during protocol generation
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"ROOT": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"ROOT": data_folder},
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"ROOT": data_folder},
    )

    # === Dataloader behaviour for TRAIN and VALID data ===
    # Step[1]: Define which column should be readed from the .csv protocol
    @sb.utils.data_pipeline.takes("Path", "Label", \
                                  "SampleRate", "AudioChannel")
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
            segment_length = random.randint(10 * int(SampleRate), \
                                            13 * int(SampleRate))
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

    # Desired datasets to use audio_pipeline()
    sb.dataio.dataset.add_dynamic_item([train_data, valid_data], audio_pipeline)
    # Desired dataset to use test_audio_pipeline()
    sb.dataio.dataset.add_dynamic_item([test_data], test_audio_pipeline)
    # Desired dataset to return pre-defined keys batch-wisely
    sb.dataio.dataset.set_output_keys(
        datasets=[train_data, valid_data, test_data], 
        output_keys=["id", "wav", "logit"]
    ) 

    # Using DynamicBatch Sampler for training data, batch size is not fixed during training,
    # and each mini-batch will contain audios with similar length
    # and total duration of each mini-batch will be limited to hparams["max_batch_length"]
    # For example, when max_batch_length is set to 22 (seconds)
    # batch-1 will have audios with length [11.1, 9.9]
    # and batch-2 will have [3.9, 4.1, 3.8, 4.2, 5.3]
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
    # We do not need test_loader_kwargs{}
    
    return train_data, valid_data, test_data, \
           train_loader_kwargs, valid_loader_kwargs

def main():
    ######
    # initialization
    ######
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    run_opts["find_unused_parameters"] = True    
    sb.utils.distributed.ddp_init_group(run_opts)
    # load configuration file
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)
    # set arndom seed
    set_random_seed(hparams["seed"])
    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    ######
    # prepare experiment 
    ######
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    ######
    # prepare dataset
    ######
    train_data, valid_data, test_data,\
    train_loader_kwargs, valid_loader_kwargs = dataio_prepare(hparams)

    ######
    # prepare SSLBrain class
    ######
    brain = SSLBrain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    ######
    # trainig start
    ###### 
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=valid_loader_kwargs,
        progressbar=True,
    )

    ######
    # evaluation start 
    ######
    brain.evaluate(
        test_data,
        min_key="EqualErrorRate",    
    )
    return

if __name__ == "__main__":
    main()
