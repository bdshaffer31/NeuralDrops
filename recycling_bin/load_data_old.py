import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io import loadmat
import random
import numpy as np

import utils


class LabelCoder:
    def __init__(self, values):
        self.encoder, self.decoder = self.create_label_encoder(values)

    def create_label_encoder(self, values):
        unique_values = list(set(values))
        encoder = {value: idx for idx, value in enumerate(unique_values)}
        decoder = {idx: value for value, idx in encoder.items()}
        return encoder, decoder

    def __call__(self, data):
        return self.apply(data)

    def apply(self, data):
        if isinstance(data, (list, torch.Tensor)):
            # Vectorized encoding for a batch of data
            return torch.tensor([self.encoder[item] for item in data], dtype=torch.long)
        else:
            # Encode a single item
            return torch.tensor(self.encoder[data], dtype=torch.long)

    def inverse_apply(self, data):
        if isinstance(data, torch.Tensor) and data.ndim > 0:
            # Vectorized decoding for a batch of encoded data
            return [self.decoder[item.item()] for item in data]
        else:
            # Decode a single encoded item
            return self.decoder[data.item() if isinstance(data, torch.Tensor) else data]


class Normalizer:
    def __init__(self, data, feature_wise=False):
        self.mean = None
        self.std = None
        self.feature_wise = feature_wise
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
        if self.std == 0.0:
            return 0.0
        return (data - self.mean) / self.std

    def inverse_apply(self, data):
        """Inverse the normalization to get the original data scale."""
        return (data * self.std) + self.mean


class Standardizer:
    def __init__(self, data=None, feature_wise=False):
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
        if self.min == self.max:
            return 0.0
        return (data - self.min) / (self.max - self.min)

    def inverse_apply(self, data):
        """Inverse the normalization to get the original data scale."""
        if self.min is None or self.max is None:
            raise ValueError("Normalization parameters not set. Call 'set_norm' first.")
        return (data * (self.max - self.min)) + self.min


class LogTransform:
    # TODO log the data and exp the model outputs so we only get positive values

    def __init__(self):
        self.epsilon = 1e-2

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        # add 1 to stabilize?
        return torch.log(self.epsilon + x)

    def inverse_apply(self, x):
        return torch.exp(x) - self.epsilon


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def apply(self, x):
        return x

    def inverse_apply(self, x):
        return x


class ProfileDataset(Dataset):
    def __init__(
        self,
        data_dir,
        experiment_numbers=None,
        valid_solutes=None,
        valid_substrates=None,
        valid_temps=None,
        temporal_subsample=1,
        spatial_subsample=1,
        axis_symmetric=False,
        temporal_pad=64,
        dtype=torch.float32,
        use_log_transform=True,
        framerate=1 / 20,
    ):
        """
        Args:
            data_dir (str): Directory containing the .mat files.
            experiment_numbers (list of int, optional): List of experiment numbers to load (e.g., [1, 2, 3]).
                                                       If None, load all experiments.
            temporal_subsample (int): Factor to subsample the time dimension.
            spatial_subsample (int): Factor to subsample the spatial dimension (x-axis) of each profile.
        """
        self.data_dir = data_dir
        self.valid_solutes = valid_solutes
        self.valid_substrates = valid_substrates
        self.valid_temps = valid_temps
        self.temporal_subsample = temporal_subsample
        self.spatial_subsample = spatial_subsample
        self.axis_symmetric = axis_symmetric
        self.temporal_pad = temporal_pad
        self.dtype = dtype
        self.use_log_transform = use_log_transform
        self.framerate = framerate

        # List of files to load
        self.setup_files(experiment_numbers, data_dir)

        # setup data scalers and encoders
        self.setup_scalers()

        # Normalize parameters across all files
        self.data = self._load_and_normalize_parameters()
        # returns a dictionary of
        # {'filename': {'temp': val, ... 'profile': tensor}}

    def setup_files(self, experiment_numbers, data_dir):
        if experiment_numbers is not None:
            self.files = [f"Exp_{num}.mat" for num in experiment_numbers]
        else:
            self.files = [
                filename
                for filename in os.listdir(data_dir)
                if filename.startswith("Exp_") and filename.endswith(".mat")
            ]
        valid_files = []
        for file in self.files:
            data = self._load_data(file)
            if self.check_valid_dataset(data):
                valid_files.append(file)
        self.valid_files = valid_files

    def check_valid_dataset(self, data):
        temp_valid = (
            float(data["temp"]) in self.valid_temps
            if self.valid_temps is not None
            else True
        )
        solute_valid = (
            data["solute"] in self.valid_solutes
            if self.valid_solutes is not None
            else True
        )
        substrate_valid = (
            data["substrate"] in self.valid_substrates
            if self.valid_substrates is not None
            else True
        )
        return temp_valid and solute_valid and substrate_valid

    def setup_scalers(self):
        # Collect parameters to fit scalers and encoders
        profiles, temps, wt_pers, solutes, substrates, times = [], [], [], [], [], []
        for file in self.valid_files:
            data = self._load_data(file)
            profile_data = self.load_profile(data)
            profiles.append(profile_data)
            temps.append(float(data["temp"]))
            wt_pers.append(float(data["wt_per"]))
            solutes.append(data["solute"])
            substrates.append(data["substrate"])
            times.append(self.setup_t_lin(len(profile_data)))

        self.profile_scaler = Standardizer(torch.cat(profiles, dim=0))
        if self.use_log_transform:
            self.log_scaler = LogTransform()
        else:
            self.log_scaler = IdentityTransform()

        self.time_scaler = Standardizer(torch.cat(times))

        # Fit scalers and encoders
        self.temp_scaler = Standardizer(torch.tensor(temps, dtype=self.dtype))
        self.wt_per_scaler = Standardizer(torch.tensor(wt_pers, dtype=self.dtype))

        # Bidirectional encoders for categorical variables
        self.solute_encoder = LabelCoder(solutes)
        self.substrate_encoder = LabelCoder(substrates)

    def _load_and_normalize_parameters(self):
        data = {}
        for file in self.valid_files:
            file_data = self._load_data(file)
            data[file] = {
                "temp": self.temp_scaler(
                    torch.tensor(float(file_data["temp"]), dtype=self.dtype)
                ),
                "wt_per": self.wt_per_scaler(
                    torch.tensor(float(file_data["wt_per"]), dtype=self.dtype)
                ),
                "solute": self.solute_encoder(file_data["solute"]),
                "substrate": self.substrate_encoder(file_data["substrate"]),
                "reflection_flag": torch.tensor(
                    float(file_data["reflection_flag"]), dtype=self.dtype
                ),
                "profile": self.log_scaler(
                    self.profile_scaler(self.load_profile(file_data))
                ),
            }
            n_samples = len(data[file]["profile"])
            t_lin = self.setup_t_lin(n_samples)
            data[file]["t"] = self.time_scaler(t_lin)  # setup t lin
            data[file]["n_samples"] = len(data[file]["profile"])
        return data

    def un_norm_data(self, data):
        return self.profile_scaler.inverse_apply(self.log_scaler.inverse_apply(data))

    def setup_t_lin(self, n_samples):
        return torch.linspace(0, n_samples * (self.framerate), n_samples)

    def load_profile(self, data):
        # put all preprocessing to load profile data here
        # leaving as individual
        np_profile = data["profile"]
        np_profile = np.array(np_profile, dtype="float32")
        profile = torch.tensor(np_profile, dtype=torch.float32)
        # apply detrending, centering, padding here
        profile, _ = utils.detrend_dataset(profile, last_n=50, window_size=50)
        profile = utils.smooth_profile(profile)
        profile = utils.pad_profile(
            profile, self.temporal_pad * self.temporal_subsample
        )
        profile = utils.vertical_crop_profile(profile, 0.78)
        profile = utils.center_data(profile)
        # if we want the axis symmetric data (half of the profile only)
        if self.axis_symmetric:
            profile = utils.central_split_data(profile)

        profile = profile[:: self.temporal_subsample, :: self.spatial_subsample]
        return profile

    def _load_data(self, file):
        contents = loadmat(os.path.join(self.data_dir, file))
        real_contents = contents["answer"]
        contents_dict = {
            "solute": real_contents[0][0][0],
            "wt_per": real_contents[1][0][0],
            "temp": real_contents[2][0][0],
            "substrate": real_contents[3][0][0],
            "reflection_flag": real_contents[4][0][0],
            "profile": real_contents[5][0].T,
        }
        if len(contents_dict["profile"].shape) < 2:
            contents_dict["profile"] = real_contents[7][0].T
        return contents_dict

    # TODO add an option for sine encoding
    def get_conditioning(self, file):
        conditioning = torch.tensor(
            [
                self.data[file]["temp"],
                self.data[file]["wt_per"],
                self.data[file]["solute"],
                self.data[file]["substrate"],
                self.data[file]["reflection_flag"],
            ],
            dtype=self.dtype,
        )
        return conditioning

    def get_input_array(self, file, t):
        # Get profile and conditioning variables as concatenated input array, applying subsampling
        profile = self.data[file]["profile"][t]
        conditioning = self.get_conditioning(file, t)
        return torch.cat([profile, conditioning])

    # needed to work as pytorch dataset, maybe not what this should do though
    def __len__(self):
        # Total number of time points across all files, after temporal subsampling
        total_length = sum(
            self.data[file]["profile"].shape[0] // self.temporal_subsample
            for file in self.files
        )
        return total_length


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_profile(h_x_t):
        for i in range(10):
            i = i * len(h_x_t) // 10
            plt.plot(h_x_t[i], c="dimgrey")
        plt.plot(h_x_t[-1], c="r")
        plt.show()

    data_dir = "data"
    file = "Exp_1.mat"
    contents = loadmat(os.path.join(data_dir, file))
    real_contents = contents["answer"]
    data = {
        "solute": real_contents[0][0][0],
        "wt_per": real_contents[1][0][0],
        "temp": real_contents[2][0][0],
        "substrate": real_contents[3][0][0],
        "reflection_flag": real_contents[4][0][0],
        "profile": real_contents[5][0].T,
    }
    np_profile = data["profile"]
    np_profile = np.array(np_profile, dtype="float32")
    profile = torch.tensor(np_profile, dtype=torch.float32)
    # apply detrending, centering, padding here
    profile, _ = utils.detrend_dataset(profile, last_n=50, window_size=50)
    profile = utils.center_data(profile)
    profile = utils.pad_profile_to_length(profile, 8000)

    profile = profile[::1, ::1]
    profile = profile - torch.min(profile)
    profile = profile / torch.max(profile)

    print(profile.shape)
    # plot_profile(profile)

    def id_drop_boundaries(h, threshold=1.0):
        # this might not be the way to do it
        threshold = torch.mean(h[:100]) * 2.0
        bool_mask = h >= threshold
        idxs = bool_mask.nonzero()
        return idxs[0][0], idxs[-1][0]

    print(id_drop_boundaries(profile[0]))
    boundaries = id_drop_boundaries(profile[0])
    print(np.array(boundaries))
    plt.plot(profile[0], c="dimgrey")
    plt.scatter(np.array(boundaries), [0, 0], c="r")
    plt.show()