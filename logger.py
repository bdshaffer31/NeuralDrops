import os
import json
import torch
import h5py
import logging
from datetime import datetime
import numpy as np


class ExperimentLogger:
    def __init__(
        self, save_dir="experiments", run_dir="run", use_timestamp=True, config=None
    ):
        # Set up a unique directory for each run
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"{run_dir}_{timestamp}"
        self.save_dir = os.path.join(save_dir, run_dir)
        self.def_name_idx = 0
        self.best_metric = None
        os.makedirs(self.save_dir, exist_ok=True)

        self.setup_logging()

        if config:
            self.log_config(config)

        logging.info("Experiment Logger initialized.")
        self.metrics_file_path = os.path.join(self.save_dir, "metrics.npz")
        self.metrics = {"epoch": []}

    def setup_logging(self, level=logging.INFO):
        # Set up logging to both file and console
        log_file_path = os.path.join(self.save_dir, "experiment.log")
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file_path),
                # logging.StreamHandler()  # Prints logs to console as well
            ],
        )

    def log_config(self, config, config_name="config.json"):
        file_path = os.path.join(self.save_dir, config_name)
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)
        logging.info("Configuration saved.")

    def log_metrics(self, metrics, epoch):
        # Append metrics to the in-memory dictionary
        self.metrics["epoch"].append(epoch)
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        np.savez(self.metrics_file_path, **self.metrics)

    def load_metrics(self):
        return dict(np.load(self.metrics_file_path))

    def save_checkpoint(self, model, optimizer, epoch, file_name="checkpoint.pth"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        file_path = os.path.join(self.save_dir, file_name)
        torch.save(checkpoint, file_path)
        logging.info(f"Checkpoint saved at epoch {epoch}.")

    def save_model(self, model, file_name="model.pth"):
        file_path = os.path.join(self.save_dir, file_name)
        torch.save(model.state_dict(), file_path)

    def save_best_model(self, model, metric, file_name="best_model.pth"):
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            self.save_model(model, file_name)
            logging.info("Best model saved.")

    def save_h5(self, data, file_name="results.h5"):
        file_path = os.path.join(self.save_dir, file_name)
        with h5py.File(file_path, "a") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
        logging.info("Data saved in HDF5 format.")

    def show(self, input_plot, ext=".png", file_name=None):
        if file_name is None:
            file_name = self.get_def_name()
        rel_filepath = self.get_relpath(f"{file_name}{ext}")
        input_plot.savefig(rel_filepath, bbox_inches="tight", transparent=False)
        input_plot.clf()
        input_plot.close()

    def show_anim(self, input_anim, ext=".gif", file_name=None):
        if file_name is None:
            file_name = self.get_def_name()
        rel_filepath = self.get_relpath(f"{file_name}{ext}")
        input_anim.save(rel_filepath, writer="ffmpeg")

    def get_def_name(self):
        file_name = f"img_{self.def_name_idx}"
        self.def_name_idx += 1
        return file_name

    def get_relpath(self, fn):
        return os.path.join(self.save_dir, fn)


class LogLoader:
    def __init__(self, run_dir, save_dir="experiments", config=None):
        # Set up a unique directory for each run
        self.save_dir = os.path.join(save_dir, run_dir)
        self.def_name_idx = 0

    def get_relpath(self, fn):
        return os.path.join(self.save_dir, fn)

    def load_metrics(self):
        metrics_path = self.get_relpath("metrics.npz")
        return dict(np.load(metrics_path))

    def load_config(self):
        config_path = self.get_relpath("config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def show(self, input_plot, ext=".png", file_name=None):
        if file_name is None:
            file_name = self.get_def_name()
        rel_filepath = self.get_relpath(f"{file_name}{ext}")
        input_plot.savefig(rel_filepath, bbox_inches="tight", transparent=False)
        input_plot.clf()
        input_plot.close()

    def show_anim(self, input_anim, ext=".gif", file_name=None):
        if file_name is None:
            file_name = self.get_def_name()
        rel_filepath = self.get_relpath(f"{file_name}{ext}")
        input_anim.save(rel_filepath, writer="ffmpeg")

    def get_def_name(self):
        file_name = f"img_{self.def_name_idx}"
        self.def_name_idx += 1
        return file_name
