import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split


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


def load_first_sheet_data():
    all_data_dict = load_drop_data_from_xlsx()
    first_dataset = all_data_dict["0.5wt% 20C"]
    data_inds = [1, 2, -1]
    data_labels = ["Radius", "Height", "Contact Angle"]
    dataset = first_dataset[1:-20, data_inds]
    return dataset, data_labels


class Normalizer:
    def __init__(self, data=None):
        self.mean = None
        self.std = None
        if data is not None:
            self.set_norm(data)

    def set_norm(self, data):
        """Set the normalization parameters based on the provided data."""
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

    def apply(self, data):
        """Apply normalization to the data using stored mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalization parameters not set. Call 'set_norm' first.")
        return (data - self.mean) / self.std

    def inverse_apply(self, data):
        """Inverse the normalization to get the original data scale."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalization parameters not set. Call 'set_norm' first.")
        return (data * self.std) + self.mean
    
class Standardizer:
    def __init__(self, data=None):
        self.max = None
        self.min = None
        if data is not None:
            self.set_norm(data)

    def set_norm(self, data):
        """Set the normalization parameters based on the provided data."""
        # hack to handle y_traj fix later
        if len(data.shape) == 3:
            self.max = torch.max(data, dim=(0,1), keepdim=True)[0]
            self.min = torch.min(data, dim=(0,1), keepdim=True)[0]
        else:
            self.max = torch.max(data, dim=0)[0]
            self.min = torch.min(data, dim=0)[0]

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


def setup_dataloaders(x, y, test_size=0.2, batch_size=32):
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    normalizer = Standardizer(X_train)

    X_train = normalizer.apply(X_train)
    X_val = normalizer.apply(X_val)
    y_train = normalizer.apply(y_train)
    y_val = normalizer.apply(y_val)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader, normalizer


if __name__ == "__main__":
    tensors = load_drop_data_from_xlsx()
    for key in tensors:
        print(f"{key}, shape: {tensors[key].shape}")
