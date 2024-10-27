import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io

import os
import json
import torch
import shutil
import csv
import h5py
import logging
from datetime import datetime

class ExperimentLogger:
    def __init__(self, save_dir, run_dir='run', use_timestamp=True, config=None):
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
            ]
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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
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

class LogLoader():
    def __init__(self, save_dir, run_dir, config=None):
        # Set up a unique directory for each run
        self.save_dir = os.path.join(save_dir, run_dir)

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


def load_drop_data_from_xlsx(excel_file="data\Qualifying Exam Data.xlsx"):
    """Load in the drop data into a dictionary of tensors"""
    sheets = pd.read_excel(excel_file, sheet_name=None)
    tensors = {}
    for sheet_name, df in sheets.items():
        if sheet_name in ["Temp Profiles", "wt% Profiles"]:
            skip_rows = 3
        else:
            skip_rows = 1
        data_numpy = np.array(
            df.values[
                skip_rows:,
                :,
            ],
            dtype=np.float64,
        )
        data_tensor = torch.tensor(data_numpy, dtype=torch.float32)
        tensors[sheet_name] = data_tensor

    return tensors

def load_test_mat_file(filename="data\Test_Drop_Boundary.mat"):
    mat_data = scipy.io.loadmat(filename)
    data = mat_data['bound']
    numpy_data = np.array([frame[:,1] for frame in data[0]], dtype='float32')
    processed_data = (numpy_data - np.min(numpy_data)) / (np.max(numpy_data) - np.min(numpy_data))
    torch_data = torch.tensor(processed_data)
    return torch_data

def detrend_dataset(dataset, window_size=10):
    # Compute the average profile across all images (along the image stack dimension)
    mean_profile = torch.mean(dataset, dim=0)
    profile_left = mean_profile[:window_size]
    
    # Fit a linear trend to the mean profile
    x = torch.arange(profile_left.shape[0], dtype=torch.float32)
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    params = torch.linalg.lstsq(A, profile_left[:, None]).solution

    # project the linear trend onto the entire x range
    x_full = torch.arange(dataset.shape[1], dtype=torch.float32)
    A_full = torch.stack([x_full, torch.ones_like(x_full)], dim=1)
    linear_trend = (A_full @ params).flatten()
    
    # Subtract the linear trend from each image in the stack
    detrended_dataset = dataset - (linear_trend[:]) # not sure why divide by 2

    return detrended_dataset, linear_trend

def end_pad_dataset(dataset, n_pad, force_zero=False):
    # append the last tensor in dataset n_pad times to the end
    # or if force_zero is true just append zeros in the shape of the data

    if force_zero:
        padding = torch.zeros((n_pad, dataset.shape[1]), dtype=dataset.dtype)
    else:
        padding = dataset[-1].repeat(n_pad, 1)
    padded_dataset = torch.cat([dataset, padding], dim=0)
    return padded_dataset

def center_data(dataset, mode='max'):
    # for a 2d tensor dataset,
    # find the x index which corresponds to the maximum y value
    # find the distance from this max x value and the x value for the center of the data
    # shift the data so the max is in the middle and pad out with the last value to keep the same shape
    max_value, max_index = torch.max(dataset, dim=1)  # max across rows (along y-axis)
    center_index = dataset.shape[1] // 2
    shift = center_index - max_index
    shift = torch.median(shift.int())
    shifted_dataset = torch.roll(dataset, shifts=int(shift), dims=1)
    
    # Padding the data to keep the same shape
    if shift > 0:
        # If we shifted the data to the right, pad the beginning with the first row
        padding = shifted_dataset[0:1].repeat(shift, 1)
        shifted_dataset = torch.cat([padding, shifted_dataset[:dataset.shape[0]-shift]], dim=0)
    elif shift < 0:
        # If we shifted the data to the left, pad the end with the last row
        padding = shifted_dataset[-1:].repeat(-shift, 1)
        shifted_dataset = torch.cat([shifted_dataset[-shift:], padding], dim=0)
    return shifted_dataset


def load_first_sheet_data():
    all_data_dict = load_drop_data_from_xlsx()
    first_dataset = all_data_dict["0.5wt% 20C"]
    data_inds = [1, 2, -1]
    data_labels = ["Radius", "Height", "Contact Angle"]
    dataset = first_dataset[1:-20, data_inds]
    return dataset, data_labels


class Normalizer:
    def __init__(self, data):
        self.mean = None
        self.std = None
        self.set_norm(data)

    def set_norm(self, data):
        """Set the normalization parameters based on the provided data."""
        if self.feature_wise:
            # Compute mean and std for each feature (dim=0)
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
        else:
            # Compute global mean and std over the entire dataset
            self.mean = torch.mean(data)
            self.std = torch.std(data)
    
    def __call__(self, data):
        return self.apply(data)

    def apply(self, data):
        """Apply normalization to the data using stored mean and std."""
        return (data - self.mean) / self.std

    def inverse_apply(self, data):
        """Inverse the normalization to get the original data scale."""
        return (data * self.std) + self.mean
    
class Standardizer:
    def __init__(self, data=None, feature_wise=True):
        self.max = None
        self.min = None
        self.feature_wise = feature_wise
        if data is not None:
            self.set_norm(data)

    def set_norm(self, data):
        """Set the normalization parameters based on the provided data."""
        if self.feature_wise:
            # Compute mean and std for each feature (dim=0)
            self.max = torch.max(data, dim=0)[0]
            self.min = torch.min(data, dim=0)[0]
        else:
            # Compute global mean and std over the entire dataset
            self.max = torch.max(data)
            self.min = torch.min(data)
    
    def __call__(self, data):
        return self.apply(data)

    def apply(self, data):
        """Apply normalization to the data using stored mean and std."""
        if self.min is None or self.max is None:
            raise ValueError("Normalization parameters not set. Call 'set_norm' first.")
        return (data - self.min) / (self.max - self.min)

    def inverse_apply(self, data):
        """Inverse the normalization to get the original data scale."""
        if self.min is None or self.max is None:
            raise ValueError("Normalization parameters not set. Call 'set_norm' first.")
        return (data * (self.max - self.min)) + self.min


def setup_dataloaders(x, y, test_size=0.2, batch_size=32, feature_norm=True):
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    normalizer = Standardizer(X_train, feature_norm)

    X_train = normalizer(X_train)
    X_val = normalizer(X_val)
    y_train = normalizer(y_train)
    y_val = normalizer(y_val)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, normalizer


if __name__ == "__main__":
    # tensor_dict = load_drop_data_from_xlsx()
    # for key in tensor_dict:
    #     print(f"{key}, shape: {tensor_dict[key].shape}")
    boundary_tensor = load_test_mat_file()
    print(boundary_tensor.shape, torch.max(boundary_tensor), torch.min(boundary_tensor), torch.mean(boundary_tensor), torch.std(boundary_tensor))
    import matplotlib.pyplot as plt
    for i in range(0, len(boundary_tensor), 100):
        plt.plot(boundary_tensor[i], c='dimgrey', alpha=((len(boundary_tensor)/2)+1+i)/(2*len(boundary_tensor)))
    plt.show()

    detreneded_bv, linear_trend = detrend_dataset(boundary_tensor, window_size=150)
    for i in range(0, len(boundary_tensor), 100):
        plt.plot(detreneded_bv[i], c='dimgrey', alpha=((len(detreneded_bv)/2)+1+i)/(2*len(detreneded_bv)))
    plt.show()

    for i in range(0, len(boundary_tensor), 100):
        plt.plot(boundary_tensor[i], c='dimgrey', alpha=((len(boundary_tensor)/2)+1+i)/(2*len(boundary_tensor)))
    # Optionally plot the linear trend
    plt.plot(linear_trend.numpy(), label='Linear Trend', color='red')
    plt.title('Computed Linear Trend')
    plt.show()

    # not quite right messes up dimensions
    shifted_bv = center_data(detreneded_bv)
    print(shifted_bv.shape)
    for i in range(0, len(shifted_bv), 100):
        plt.plot(shifted_bv[i], c='dimgrey', alpha=((len(shifted_bv)/2)+1+i)/(2*len(shifted_bv)))
    plt.show()