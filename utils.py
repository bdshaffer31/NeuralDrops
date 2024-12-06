import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io
import load_data
from scipy.ndimage import median_filter, gaussian_filter

def load_single_mat_file(filename):
    contents = scipy.io.loadmat(filename)
    real_contents = contents["answer"]
    contents_dict = {
        "solute": real_contents[0][0],
        "wt_per": real_contents[1][0],
        "temp": real_contents[2][0],
        "substrate": real_contents[3][0],
        "reflection_flag": real_contents[4][0],
        "profile": real_contents[5][0].T,
    }
    return contents_dict


def create_trajectories(x, traj_len=32):
    """
    Create training data where each time step acts as an initial condition,
    and the subsequent points in the trajectory form the ground truth, with variable lengths.
    """
    x_init = []
    y_traj = []

    for i in range(len(x) - traj_len):
        x_init.append(x[i])
        y_traj.append(x[i + 1 : i + traj_len + 1])

    x_init = torch.stack(x_init)
    y_traj = torch.stack(y_traj)

    return x_init, y_traj


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
    data = mat_data["bound"]
    numpy_data = np.array([frame[:, 1] for frame in data[0]], dtype="float32")
    processed_data = (numpy_data - np.min(numpy_data)) / (
        np.max(numpy_data) - np.min(numpy_data)
    )
    torch_data = torch.tensor(processed_data)
    return torch_data


def detrend_dataset(dataset, last_n=100, window_size=10):
    # Compute the average profile across all images (along the image stack dimension)
    last_data = dataset[-last_n:]
    mean_profile = torch.mean(last_data, dim=0)
    profile_left = mean_profile[:window_size]
    profile_right = mean_profile[-window_size:]

    # Concatenate the left and right profiles
    x_left = torch.arange(profile_left.shape[0], dtype=torch.float32)
    x_right = torch.arange(
        mean_profile.shape[0] - window_size, mean_profile.shape[0], dtype=torch.float32
    )
    x = torch.cat([x_left, x_right])
    profile_combined = torch.cat([profile_left, profile_right])

    # Fit a linear trend to the mean profile
    A = torch.stack([x, torch.ones_like(x)], dim=1)
    params = torch.linalg.lstsq(A, profile_combined[:, None]).solution

    # project the linear trend onto the entire x range
    x_full = torch.arange(dataset.shape[1], dtype=torch.float32)
    A_full = torch.stack([x_full, torch.ones_like(x_full)], dim=1)
    linear_trend = (A_full @ params).flatten()

    # Subtract the linear trend from each image in the stack
    detrended_dataset = dataset - (linear_trend[:])  # not sure why divide by 2

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


def pad_profile_to_length(profile, target_length):
    current_length = profile.shape[0]

    if current_length >= target_length:
        return profile[:target_length]

    num_frames_to_add = target_length - current_length

    last_frame = profile[-1:]
    padding_frames = last_frame.repeat(num_frames_to_add, *[1] * (profile.dim() - 1))

    padded_profile = torch.cat([profile, padding_frames], dim=0)
    return padded_profile


def center_data(dataset, mode="max"):
    # Find the x index corresponding to the maximum y value in each row
    max_values, max_indices = torch.max(dataset, dim=1)

    # Calculate the shift needed to center the max values
    center_index = dataset.shape[1] // 2
    shifts = center_index - max_indices
    median_shift = int(
        torch.median(shifts).item()
    )  # Get the median shift to apply uniformly

    # Shift the dataset by the median shift amount
    shifted_dataset = torch.roll(dataset, shifts=median_shift, dims=1)

    # Fill missing values created by the shift with the closest known values
    if median_shift > 0:
        # If we shifted right, fill the beginning with the first column
        padding = shifted_dataset[:, 0:1].repeat(1, median_shift)
        shifted_dataset[:, :median_shift] = padding
    elif median_shift < 0:
        # If we shifted left, fill the end with the last column
        padding = shifted_dataset[:, -1:].repeat(1, -median_shift)
        shifted_dataset[:, median_shift:] = padding

    return shifted_dataset

def smooth_profile(profile, filter_size=40, sigma=10, temporal_filter=5, temporal_sigma=2):
    profile_np = profile.numpy()
    # filter / smooth in space
    filtered_profile = median_filter(profile_np, size=(1, filter_size)) # time filter, space filter
    smoothed_profile = gaussian_filter(filtered_profile, sigma=(0, sigma)) # time sigma, space sigma
    # filter / smooth in time
    filtered_profile = median_filter(smoothed_profile, size=(temporal_filter, 1))
    smoothed_profile = gaussian_filter(filtered_profile, sigma=(temporal_sigma, 0))
    processed_profile = torch.tensor(smoothed_profile, dtype=profile.dtype)
    return processed_profile

def vertical_crop_profile(profile, threshold=0.8):
    return torch.max(torch.zeros_like(profile), profile - threshold*torch.max(profile[-1]))

def pad_profile(profile, pad=32):
    last_frame = profile[-1:]
    padding_frames = last_frame.repeat(pad, *[1] * (profile.dim() - 1))
    padded_profile = torch.cat([profile, padding_frames], dim=0)
    return padded_profile

def load_first_sheet_data():
    all_data_dict = load_drop_data_from_xlsx()
    first_dataset = all_data_dict["0.5wt% 20C"]
    data_inds = [1, 2, -1]
    data_labels = ["Radius", "Height", "Contact Angle"]
    dataset = first_dataset[1:-20, data_inds]
    return dataset, data_labels


def setup_dataloaders(x, y, test_size=0.2, batch_size=32, feature_norm=True):
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    normalizer = load_data.Standardizer(X_train, feature_norm)

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

    load_data = load_single_mat_file("data/Exp_1.mat")
    for key, val in load_data.items():
        print(key, val)
    print(load_data["profile"].shape)
    input()

    boundary_tensor = load_test_mat_file()
    print(
        boundary_tensor.shape,
        torch.max(boundary_tensor),
        torch.min(boundary_tensor),
        torch.mean(boundary_tensor),
        torch.std(boundary_tensor),
    )
    import matplotlib.pyplot as plt

    for i in range(0, len(boundary_tensor), 100):
        plt.plot(
            boundary_tensor[i],
            c="dimgrey",
            alpha=((len(boundary_tensor) / 2) + 1 + i) / (2 * len(boundary_tensor)),
        )
    plt.show()

    detreneded_bv, linear_trend = detrend_dataset(boundary_tensor, window_size=150)
    for i in range(0, len(boundary_tensor), 100):
        plt.plot(
            detreneded_bv[i],
            c="dimgrey",
            alpha=((len(detreneded_bv) / 2) + 1 + i) / (2 * len(detreneded_bv)),
        )
    plt.show()

    for i in range(0, len(boundary_tensor), 100):
        plt.plot(
            boundary_tensor[i],
            c="dimgrey",
            alpha=((len(boundary_tensor) / 2) + 1 + i) / (2 * len(boundary_tensor)),
        )
    # Optionally plot the linear trend
    plt.plot(linear_trend.numpy(), label="Linear Trend", color="red")
    plt.title("Computed Linear Trend")
    plt.show()

    # not quite right messes up dimensions
    shifted_bv = center_data(detreneded_bv)
    print(shifted_bv.shape)
    for i in range(0, len(shifted_bv), 100):
        plt.plot(
            shifted_bv[i],
            c="dimgrey",
            alpha=((len(shifted_bv) / 2) + 1 + i) / (2 * len(shifted_bv)),
        )
    plt.show()
