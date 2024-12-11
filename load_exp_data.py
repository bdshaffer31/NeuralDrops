import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import numpy as np


from utils import detrend_dataset, smooth_profile, pad_profile, vertical_crop_profile, center_data

class ExperimentDataLoader:
    def __init__(self, data_dir, experiment_numbers=None):
        self.data_dir = data_dir
        self.experiment_numbers = experiment_numbers
        self.files = self._get_files()

    def _get_files(self):
        if self.experiment_numbers is not None:
            return [f"Exp_{num}.mat" for num in self.experiment_numbers]
        return [
            filename for filename in os.listdir(self.data_dir)
            if filename.startswith("Exp_") and filename.endswith(".mat")
        ]

    def load_data(self, file):
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

class ExperimentTransformer:
    def __init__(self, use_log_transform=False, temporal_pad=64, temporal_subsample=1, spatial_subsample=1, axis_symmetric=False, dtype=torch.float32):
        self.use_log_transform = use_log_transform
        self.temporal_pad = temporal_pad
        self.temporal_subsample = temporal_subsample
        self.spatial_subsample = spatial_subsample
        self.axis_symmetric = axis_symmetric
        self.dtype = dtype

    def transform_profile(self, profile):
        profile = np.array(profile, dtype="float32")
        profile = torch.tensor(profile, dtype=torch.float32)
        profile, _ = detrend_dataset(profile, last_n=50, window_size=50)
        profile = smooth_profile(profile)
        profile = pad_profile(profile, self.temporal_pad * self.temporal_subsample)
        profile = vertical_crop_profile(profile, 0.78)
        profile = center_data(profile)

        if self.axis_symmetric:
            profile = profile[:, :profile.shape[1] // 2]

        profile = profile[::self.temporal_subsample, ::self.spatial_subsample]
        return profile

    def apply_normalization(self, profiles, temps, wt_pers, solutes, substrates, times):
        # Initialize scalers and encoders
        self.profile_scaler = Standardizer(torch.cat(profiles, dim=0))
        self.time_scaler = Standardizer(torch.cat(times))
        self.temp_scaler = Standardizer(torch.tensor(temps, dtype=self.dtype))
        self.wt_per_scaler = Standardizer(torch.tensor(wt_pers, dtype=self.dtype))

        self.solute_encoder = LabelCoder(solutes)
        self.substrate_encoder = LabelCoder(substrates)

        if self.use_log_transform:
            self.log_scaler = LogTransform()
        else:
            self.log_scaler = IdentityTransform()

    def normalize_data(self, data):
        return {
            "temp": self.temp_scaler(torch.tensor(float(data["temp"]), dtype=self.dtype)),
            "wt_per": self.wt_per_scaler(torch.tensor(float(data["wt_per"]), dtype=self.dtype)),
            "solute": self.solute_encoder(data["solute"]),
            "substrate": self.substrate_encoder(data["substrate"]),
            "reflection_flag": torch.tensor(float(data["reflection_flag"]), dtype=self.dtype),
            "profile": self.log_scaler(self.profile_scaler(self.transform_profile(data["profile"])))
        }

    def unnormalize_profile(self, profile):
        return self.profile_scaler.inverse_apply(self.log_scaler.inverse_apply(profile))


class ExperimentDataset(Dataset):
    def __init__(self, data_loader, transformer):
        self.data_loader = data_loader
        self.transformer = transformer
        self.data = self._load_and_process_data()

    def _load_and_process_data(self):
        raw_data_list = [self.data_loader.load_data(file) for file in self.data_loader.files]

        # Collect parameters for normalization
        profiles, temps, wt_pers, solutes, substrates, times = [], [], [], [], [], []
        for raw_data in raw_data_list:
            # profile = self.transformer.transform_profile(torch.tensor(raw_data["profile"], dtype=torch.float32))
            profile = self.transformer.transform_profile(raw_data["profile"])
            profiles.append(profile)
            temps.append(float(raw_data["temp"]))
            wt_pers.append(float(raw_data["wt_per"]))
            solutes.append(raw_data["solute"])
            substrates.append(raw_data["substrate"])
            times.append(torch.linspace(0, len(profile) * 1 / 20, len(profile)))

        # Apply normalization
        self.transformer.apply_normalization(profiles, temps, wt_pers, solutes, substrates, times)

        # Normalize the data
        return {file: self.transformer.normalize_data(raw_data) for file, raw_data in zip(self.data_loader.files, raw_data_list)}

    def get_conditioning(self, file):
        conditioning = torch.tensor(
            [
                self.data[file]["temp"],
                self.data[file]["wt_per"],
                self.data[file]["solute"],
                self.data[file]["substrate"],
                self.data[file]["reflection_flag"],
            ],
            dtype=self.transformer.dtype,
        )
        return conditioning

    def __len__(self):
        return sum(len(data["profile"]) for data in self.data.values())

    def __getitem__(self, idx):
        current_idx = 0
        for file, data in self.data.items():
            profile_len = len(data["profile"])
            if current_idx + profile_len > idx:
                relative_idx = idx - current_idx
                profile = data["profile"][relative_idx]
                conditioning = torch.tensor([
                    data["temp"],
                    data["wt_per"],
                    data["solute"],
                    data["substrate"],
                    data["reflection_flag"]
                ], dtype=self.transformer.dtype)
                return profile, conditioning
            current_idx += profile_len


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



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_dir = "data"
    experiment_numbers = [1, 20, 30]

    # Initialize DataLoader and Transformer
    data_loader = ExperimentDataLoader(data_dir, experiment_numbers)
    transformer = ExperimentTransformer(use_log_transform=False, temporal_pad=64, temporal_subsample=32)

    # Create Dataset
    dataset = ExperimentDataset(data_loader, transformer)

    # Access data
    for profile, conditioning in dataset:
        print(profile.shape, conditioning.shape)
        print(conditioning)
        plt.plot(profile)
        plt.show()

    # Unnormalize a profile
    unnormalized_profile = transformer.unnormalize_profile(profile)
