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

class LogTransform():
    # TODO log the data and exp the model outputs so we only get positive values

    def __init__(self):
        self.epsilon = 1e-2

    def __call__(self, x):
        return self.apply(x)
    
    def apply(self, x):
        # add 1 to stabilize?
        return torch.log(self.epsilon+x)
    
    def inverse_apply(self, x):
        return torch.exp(x)-self.epsilon

class IdentityTransform():

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
        temporal_subsample=1,
        spatial_subsample=1,
        dtype=torch.float32,
        use_log_transform=True,
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
        self.temporal_subsample = temporal_subsample
        self.spatial_subsample = spatial_subsample
        self.dtype = dtype
        self.use_log_transform = use_log_transform
        self.max_profile_length = (
            7000  # TODO actually determine this from data + some buffer
        )

        # List of files to load
        if experiment_numbers is not None:
            self.files = [f"Exp_{num}.mat" for num in experiment_numbers]
        else:
            self.files = [
                filename
                for filename in os.listdir(data_dir)
                if filename.startswith("Exp_") and filename.endswith(".mat")
            ]

        # setup data scalers and encoders
        self.setup_scalers()

        # Normalize parameters across all files
        self.data = self._load_and_normalize_parameters()
        # returns a dictionary of
        # {'filename': {'temp': val, ... 'profile': tensor}}

    def setup_scalers(self):
        # Collect parameters to fit scalers and encoders
        profiles, temps, wt_pers, solutes, substrates = [], [], [], [], []
        for file in self.files:
            data = self._load_data(file)
            profiles.append(self.load_profile(data))
            temps.append(float(data["temp"]))
            wt_pers.append(float(data["wt_per"]))
            solutes.append(data["solute"])
            substrates.append(data["substrate"])

        self.profile_scaler = Standardizer(torch.cat(profiles, dim=0))
        if self.use_log_transform:
            self.log_scaler = LogTransform()
        else:
            self.log_scaler = IdentityTransform()

        # Fit scalers and encoders
        self.temp_scaler = Standardizer(torch.tensor(temps, dtype=self.dtype))
        self.wt_per_scaler = Standardizer(torch.tensor(wt_pers, dtype=self.dtype))

        # Bidirectional encoders for categorical variables
        self.solute_encoder = LabelCoder(solutes)
        self.substrate_encoder = LabelCoder(substrates)

    def _load_and_normalize_parameters(self):
        data = {}
        for file in self.files:
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
                    self.profile_scaler(self.load_profile(file_data))),
            }
        return data

    def un_norm_data(self, data):
        return self.profile_scaler.inverse_apply(self.log_scaler.inverse_apply(data))

    def load_profile(self, data):
        np_profile = data["profile"]
        np_profile = np.array(np_profile, dtype="float32")
        profile = torch.tensor(np_profile, dtype=self.dtype)
        # apply detrending, centering, padding here
        profile, _ = utils.detrend_dataset(profile, last_n=50, window_size=50)
        profile = utils.center_data(profile)
        profile = utils.pad_profile_to_length(profile, 8000)

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
        return contents_dict

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
        conditioning = self.get_conditioning(file)
        return torch.cat([profile, conditioning])

    # needed to work as pytorch dataset, maybe not what this should do though
    def __len__(self):
        # Total number of time points across all files, after temporal subsampling
        total_length = sum(
            self.data[file]["profile"].shape[0] // self.temporal_subsample
            for file in self.files
        )
        return total_length


class NODEDataset(Dataset):
    def __init__(self, profile_data, traj_len=10):
        """
        Args:
            profile_data (ProfileDataset): Instance of ProfileDataset for data access.
            k (int): Number of future snapshots to include in the target sequence.
        """
        self.profile_data = profile_data
        self.traj_len = traj_len
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for file, data in self.profile_data.data.items():
            profile = data["profile"]
            # Generate samples with initial condition and future snapshots
            for t in range(profile.shape[0] - self.traj_len):
                initial_condition = profile[t]
                target_snapshots = profile[t + 1 : t + 1 + self.traj_len]
                conditioning = self.profile_data.get_conditioning(file)
                samples.append((initial_condition, conditioning, target_snapshots))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NeuralFieldDataset(Dataset):
    def __init__(self, profile_data, num_samples=1000):
        """
        Args:
            profile_data (ProfileDataset): Instance of ProfileDataset for data access.
            num_samples (int): Number of random (x, t) samples per experiment for training.
        """
        self.profile_data = profile_data
        self.num_samples = num_samples
        self.samples = self._generate_samples()

        example_file = list(self.profile_data.data.keys())[0]
        example_profile = self.profile_data.data[example_file]["profile"]
        self.profile_dim = example_profile.shape[
            1
        ]  # Width of the profile (x-dimension)
        self.conditioning_dim = 5

    def _generate_samples(self):
        samples = []
        for file, data in self.profile_data.data.items():
            profile = data["profile"]
            t_max, x_max = profile.shape

            conditioning = self.profile_data.get_conditioning(file)

            if self.num_samples is None:
                # Generate a sample for each (x, t) coordinate in the profile
                for t in range(t_max):
                    for x in range(x_max):
                        coordinate = torch.tensor([x, t], dtype=self.profile_data.dtype)
                        target_value = profile[t, x]
                        samples.append((conditioning, coordinate, target_value))
            else:
                # Generate `num_samples` random (x, t) coordinates
                for _ in range(self.num_samples):
                    t = random.randint(0, t_max - 1)
                    x = random.randint(0, x_max - 1)
                    coordinate = torch.tensor([x, t], dtype=self.profile_data.dtype)
                    target_value = profile[t, x]
                    samples.append((conditioning, coordinate, target_value))
        return samples

    @property
    def input_dim(self):
        # Input dimension is profile dimension + conditioning dimension
        return self.profile_dim + self.conditioning_dim

    @property
    def output_dim(self):
        # Output dimension is just the profile dimension
        return self.profile_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def setup_node_data(
    traj_len,
    batch_size,
    exp_nums=None,
    temporal_subsample=10,
    spatial_subsample=2,
    use_log_transform=True,
    data_dir="data",
    test_split=0.1,
):
    profile_data = ProfileDataset(
        data_dir=data_dir,
        experiment_numbers=exp_nums,
        temporal_subsample=temporal_subsample,
        spatial_subsample=spatial_subsample,
        dtype=torch.float32,
        use_log_transform=use_log_transform
    )

    # Create NODEDataset with the specified trajectory length
    node_dataset = NODEDataset(profile_data=profile_data, traj_len=traj_len)

    # Split dataset into training and validation sets
    dataset_size = len(node_dataset)
    val_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(node_dataset, [train_size, val_size])

    # Set up DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, profile_data
