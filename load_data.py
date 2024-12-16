# loop over experiment data
import os
import torch
from scipy.io import loadmat
import numpy as np
import utils  # Assuming your utility functions are in utils.py
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from drop_model import utils as drop_utils


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


def load_data(file_path):
    """Load the data from a .mat file and return as a dictionary."""
    contents = loadmat(file_path)
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


def preprocess_profile(
    profile, temporal_pad, temporal_subsample, spatial_subsample, axis_symmetric
):
    """Preprocess the profile data."""
    profile = np.array(profile, dtype="float64")
    profile = torch.tensor(profile, dtype=torch.float64)
    profile, _ = utils.detrend_dataset(profile, last_n=50, window_size=50)
    profile = utils.smooth_profile(profile)
    profile = utils.pad_profile(profile, temporal_pad * temporal_subsample)
    profile = utils.vertical_crop_profile(profile, 0.78)
    profile = utils.center_data(profile)
    if axis_symmetric:
        profile = utils.central_split_data(profile)
    profile = profile[::temporal_subsample, spatial_subsample // 2 :: spatial_subsample]
    return profile


def preprocess_and_save(
    data_dir,
    run_nums,
    output_file,
    valid_temps=None,
    valid_solutes=None,
    valid_substrates=None,
    temporal_pad=64,
    temporal_subsample=1,
    spatial_subsample=1,
    axis_symmetric=False,
    use_log_transform=True,
    dtype=torch.float64,
):
    """Preprocess all valid files in the data directory and save them in a single .pth file."""
    m_per_px = 0.00003

    # Find all valid files
    files = [f"Exp_{f}.mat" for f in run_nums]

    # Data lists for fitting scalers and encoders
    h_max, solutes, substrates = [], [], []

    # Load and grab the values that we need to make dictionaries for
    for file in files:
        data = load_data(os.path.join(data_dir, file))
        solutes.append(data["solute"])
        substrates.append(data["substrate"])
        h_max.append(np.max(data["profile"]))

    # Fit encoders
    solute_encoder = LabelCoder(solutes)
    substrate_encoder = LabelCoder(substrates)
    h_max = max(h_max)
    print(h_max)

    # for now just grab these for later
    print(solute_encoder.encoder)
    print(substrate_encoder.encoder)

    # Preprocess and normalize data
    preprocessed_data = {}
    for run_num, file in zip(run_nums, files):
        print(file)
        data = load_data(os.path.join(data_dir, file))
        raw_profile = data["profile"]
        t_lin = torch.arange(raw_profile.shape[0] + temporal_pad*temporal_subsample) / 20
        r_lin = torch.arange(raw_profile.shape[1]) * m_per_px
        z_lin = torch.arange(h_max) * m_per_px
        raw_profile = raw_profile * m_per_px
        profile = preprocess_profile(
            raw_profile,
            temporal_pad,
            temporal_subsample,
            spatial_subsample,
            axis_symmetric,
        )

        t_lin = t_lin[::temporal_subsample]
        r_lin = r_lin[spatial_subsample // 2 :: spatial_subsample]
        z_lin = z_lin[::spatial_subsample]

        # ensure even dimension
        if profile.shape[1] % 2 == 1: # need an odd number of values for non zero r point ... 
            # sort of janky, just drop the first value
            profile = profile[:,1:]
            r_lin = r_lin[1:]
        # force symmetric, this should be moved to process but needs to hapenn after force even
        profile = drop_utils.symmetrize(profile)

        preprocessed_data[run_num] = {
            "profile": profile,
            "t_lin": t_lin,
            "r_lin": r_lin,
            "z_lin": z_lin,
            "temp": torch.tensor(float(data["temp"]), dtype=dtype),
            "wt_per": torch.tensor(float(data["wt_per"]), dtype=dtype),
            "solute": solute_encoder(data["solute"]),
            "substrate": substrate_encoder(data["substrate"]),
        }

    # Save preprocessed data
    torch.save(preprocessed_data, output_file)
    print(f"Preprocessed data saved to {output_file}")


# TODO create a base class and inherit for these to simplify
class NODEDataset(Dataset):
    def __init__(
        self,
        data,
        conditioning_keys=None,
        traj_len=10,
        profile_key="profile",
        run_keys=None,
        profile_scale=1.0,
    ):
        """
        Args:
            data (dict): Dictionary of preprocessed data.
            traj_len (int): Number of future snapshots to include in the target sequence.
            conditioning_keys (list of str): Keys to use for conditioning variables.
            profile_key (str): Key to access the profile data in each sample.
        """
        self.run_keys = run_keys
        self.data = {k: v for k, v in data.items() if run_keys is None or k in run_keys}
        self.traj_len = traj_len
        self.conditioning_keys = conditioning_keys
        self.profile_key = profile_key
        self.profile_scale = profile_scale
        self.samples = self._prepare_samples()

    def get_conditioning(self, exp_data):
        z = torch.tensor(
            [exp_data[key] for key in self.conditioning_keys], dtype=torch.float64
        )
        return z

    def _prepare_samples(self):
        samples = []
        for exp_key, exp_data in self.data.items():
            profile = exp_data[self.profile_key]
            profile = profile * self.profile_scale
            num_time_steps = profile.shape[0]

            for t in range(num_time_steps - self.traj_len):
                h0 = profile[t]
                target_h = profile[t + 1 : t + 1 + self.traj_len]
                z = self.get_conditioning(exp_data)
                t = torch.linspace(0, self.traj_len, self.traj_len)
                t = torch.linspace(1, self.traj_len, self.traj_len)

                samples.append((t, h0, z, target_h))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FNODataset(Dataset):
    def __init__(
        self,
        data,
        conditioning_keys=None,
        profile_key="profile",
        run_keys=None,
        profile_scale=1.0,
    ):
        """
        Args:
            data (dict): Dictionary of preprocessed data.
            conditioning_keys (list of str): Keys to use for conditioning variables.
            profile_key (str): Key to access the profile data in each sample.
        """
        self.run_keys = run_keys
        self.data = {
            k: v.to(torch.float64)
            for k, v in data.items()
            if run_keys is None or k in run_keys
        }
        self.conditioning_keys = conditioning_keys
        self.profile_key = profile_key
        self.profile_scale = profile_scale
        self.samples = self._prepare_samples()

    def get_conditioning(self, exp_data):
        z = torch.tensor(
            [exp_data[key] for key in self.conditioning_keys], dtype=torch.float64
        )
        return z

    def _prepare_samples(self):
        samples = []
        for exp_key, exp_data in self.data.items():
            profile = exp_data[self.profile_key]  # Shape: [time, x]
            profile = profile * self.profile_scale
            time_steps = exp_data["t_lin"]  # Shape: [time]
            conditioning = self.get_conditioning(exp_data)

            # Create samples for each time step starting from the second time step
            for t_idx in range(1, len(time_steps)):
                h_0 = profile[0]  # Initial condition at t=0
                h_t = profile[t_idx]  # Profile at time t
                t = time_steps[t_idx].unsqueeze(-1)  # Time step as [1]

                samples.append((t, h_0, conditioning, h_t))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_train_val_dataloaders(dataset, val_ratio=0.1, batch_size=16, shuffle=True):
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def setup_data_from_config(config):
    data_file = config["data_file"]
    loaded_data = torch.load(data_file)

    if config["model_type"] in ["node", "fno_node", "flux_fno"]:
        dataset = NODEDataset(
            loaded_data,
            config["conditioning_keys"],
            traj_len=config["traj_len"],
            run_keys=config["run_keys"],
            profile_scale=config["profile_scale"],
        )
    elif config["model_type"] == "fno":
        dataset = FNODataset(
            loaded_data,
            config["conditioning_keys"],
            run_keys=config["run_keys"],
            profile_scale=config["profile_scale"],
        )

    train_loader, val_loader = get_train_val_dataloaders(
        dataset, config["val_ratio"], config["batch_size"], shuffle=True
    )

    return train_loader, val_loader, dataset


if __name__ == "__main__":
    data_dir = (
        "data/raw_drop_data"  # Replace with the actual path to your data directory
    )
    output_file = "data/drop_data_10.pth"

    preprocess_and_save(
        data_dir=data_dir,
        run_nums=utils.good_run_numbers()[:10],
        output_file=output_file,
        temporal_pad=128,
        temporal_subsample=12,
        spatial_subsample=6,
        axis_symmetric=False,
    )

    # data = torch.load("data/drop_data.pth")
    # # print(data)
    # p = data[1]["profile"]
    # print(torch.min(p), torch.max(p))

    # filename = "data/simulation_results.pth"

    # # List of keys to use as conditioning variables
    # conditioning_keys = ["alpha", "beta", "gamma"]
    # # conditioning_keys = ["temp", "wt_per", "solute"]

    # # Create the DataLoader
    # dataloader = get_dataloader(
    #     filename, conditioning_keys, batch_size=16, shuffle=True
    # )

    # # Iterate through the DataLoader
    # for profile, conditioning in dataloader:
    #     print("Profile shape:", profile.shape)
    #     print("Conditioning shape:", conditioning.shape)
    #     break
