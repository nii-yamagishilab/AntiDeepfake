"""Optional W&B tracking helpers for main_try.py (disabled by default)."""
import argparse
import contextlib
import functools
import json
import logging
import os
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_wandb_arguments(arguments):
    """Consume W&B flags and leave SpeechBrain arguments untouched."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "AntiDeepfake"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.getenv("WANDB_NAME"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    options, remaining = parser.parse_known_args(arguments)
    if not remaining:
        parser.error("Provide a HyperPyYAML file, optionally preceded by inference or analysis")
    return options, remaining


def wandb_session(function):
    """Close tracking contexts on success or failure without indenting main()."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            return function(stack, *args, **kwargs)
    return wrapped


def config_value(value):
    """Serialize resolved HyperPyYAML values without traversing model weights."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_value(v) for v in value]
    if isinstance(value, functools.partial):
        return {"callable": config_value(value.func),
                "args": config_value(value.args),
                "kwargs": config_value(value.keywords)}
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__module__}.{type(value).__qualname__}"


class Tee:
    """Keep console output visible while writing it to a local file."""
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        return self.stream.write(text)

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class WandbLogger:
    """Forward existing SpeechBrain logs to W&B without replacing local logs."""
    def __init__(self, original, run):
        self.original, self.run = original, run

    def log_stats(self, stats_meta, train_stats=None, valid_stats=None,
                  test_stats=None, **kwargs):
        self.original.log_stats(stats_meta=stats_meta, train_stats=train_stats,
                                valid_stats=valid_stats, test_stats=test_stats,
                                **kwargs)
        metrics = dict(stats_meta)
        for stage, values in (("train", train_stats), ("valid", valid_stats),
                              ("test", test_stats)):
            if values:
                metrics.update({f"{stage}/{k}": v for k, v in values.items()})
        # Use an explicit axis, not W&B's internal step: validation and training
        # can both report the same optimizer step.
        self.run.log(metrics)

    def __getattr__(self, name):
        return getattr(self.original, name)


class RunTracker:
    def __init__(self, run, wandb):
        self.run, self.wandb = run, wandb
        self.last_checkpoint = None

    def save_file(self, path):
        """Upload an output file to the run's Files tab, keeping its filename."""
        path = Path(path).resolve()
        self.run.save(str(path), base_path=str(path.parent), policy="now")

    def checkpoint(self, checkpointer):
        checkpoint = checkpointer.find_checkpoint(min_key="EqualErrorRate")
        if checkpoint is None or str(checkpoint.path) == self.last_checkpoint:
            return
        artifact = self.wandb.Artifact(
            f"checkpoint-{self.run.id}", type="model",
            metadata=config_value(checkpoint.meta))
        # Include optimizer, scheduler, brain and loader state as well as weights.
        artifact.add_dir(str(checkpoint.path))
        self.run.log_artifact(artifact, aliases=["best", "latest"])
        self.last_checkpoint = str(checkpoint.path)


@contextlib.contextmanager
def track_run(options, hparams, hparams_file, overrides, run_opts, working_mode):
    """Only rank zero creates a run and uploads shared experiment files."""
    if options.wandb_mode == "disabled":
        yield None
        return
    from speechbrain.utils.distributed import if_main_process

    if not if_main_process():
        yield None
        return
    output = Path(hparams["output_folder"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {"hparams": config_value(hparams), "run_options": config_value(run_opts),
              "overrides": config_value(overrides), "working_mode": working_mode}
    config_path = output / "wandb_hparams.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    run = tracker = None
    if options.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install wandb with: pip install wandb; or use "
                               "--wandb-mode disabled") from exc
        run = wandb.init(project=options.wandb_project, entity=options.wandb_entity,
                         name=options.wandb_name, mode=options.wandb_mode,
                         dir=str(output), config=config, job_type=working_mode)
        run.define_metric("steps")
        run.define_metric("*", step_metric="steps")
        tracker = RunTracker(run, wandb)
        for key in ("train_steps_logger", "train_stage_logger"):
            if key in hparams:
                hparams[key] = WandbLogger(hparams[key], run)
    exit_code = 0
    try:
        with (output / "output.log").open("a", encoding="utf-8", buffering=1) as log:
            stdout, stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
            # Existing logging handlers retain their original stream references.
            handlers = []
            loggers = [logging.getLogger()] + [
                item for item in logging.Logger.manager.loggerDict.values()
                if isinstance(item, logging.Logger)]
            try:
                for item in loggers:
                    for handler in item.handlers:
                        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                            original_stream = handler.stream
                            try:
                                handler.setStream(stdout if original_stream is sys.stdout else stderr)
                            except AttributeError:
                                # Some handlers expose a read-only stream property
                                # that follows sys.stderr and needs no reassignment.
                                continue
                            handlers.append((handler, original_stream))
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        yield tracker
                    except BaseException:
                        traceback.print_exc()
                        raise
            finally:
                for handler, stream in handlers:
                    handler.setStream(stream)
    except BaseException:
        exit_code = 1
        raise
    finally:
        if run is not None:
            try:
                # Upload files individually: never recursively upload datasets or
                # pretrained models referenced by the experiment configuration.
                files = {output / "output.log", config_path,
                         output / "hyperparams.yaml", output / "log.txt",
                         Path(hparams_file).resolve(), Path(__file__).resolve(),
                         Path(__file__).with_name("main_try.py").resolve()}
                for key in ("train_steps_log", "train_stage_log"):
                    if hparams.get(key):
                        files.add(Path(hparams[key]).resolve())
                for file in sorted(files):
                    if file.is_file():
                        run.save(str(file), base_path=str(file.parent), policy="now")
            except Exception:
                logger.exception("Could not upload final W&B log files; local files remain available")
            finally:
                run.finish(exit_code=exit_code)

