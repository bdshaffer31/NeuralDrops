import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random


class NODEDataset(Dataset):
    def __init__(self, profile_data, traj_len=10):
        """
        Args:
            profile_data (ProfileDataset): Instance of ProfileDataset for data access.
            k (int): Number of future snapshots to include in the target sequence.
        """
        self.profile_data = profile_data
        self.traj_len = traj_len
        self.time_steps = self.get_time_steps()
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
                time_steps = self.time_steps
                samples.append(
                    (time_steps, initial_condition, conditioning, target_snapshots)
                )
        return samples

    def get_time_steps(self):
        return self.profile_data.time_scaler(
            self.profile_data.setup_t_lin(self.traj_len)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class FNODataset(Dataset):
    def __init__(self, profile_data, num_time_points=10):
        self.profile_data = profile_data
        self.num_time_points = num_time_points
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for file, data in self.profile_data.data.items():
            profile = data["profile"]
            time_steps = data["t"]
            conditioning = self.profile_data.get_conditioning(file)

            # Create samples for each time step
            for t_idx in range(1, len(time_steps)):
                h_0 = profile[0]  # Initial condition
                h_t = profile[t_idx]  # Profile at time t
                t = time_steps[t_idx].unsqueeze(-1)  # Time
                samples.append((h_0, conditioning, t, h_t))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h_0, z, t, h_t = self.samples[idx]
        return h_0, z, t, h_t


# def setup_data(config):
#     profile_data = ProfileDataset(
#         data_dir=config["data_dir"],
#         experiment_numbers=config["exp_nums"],
#         valid_solutes=config["valid_solutes"],
#         valid_substrates=config["valid_substrates"],
#         valid_temps=config["valid_temps"],
#         temporal_subsample=config["temporal_subsample"],
#         spatial_subsample=config["spatial_subsample"],
#         axis_symmetric=config["axis_symmetric"],
#         temporal_pad=config["temporal_pad"],
#         dtype=torch.float32,
#         use_log_transform=config["use_log_transform"],
#     )

#     if config["model_type"] in ["node", "flux_fno"]:
#         dataset = NODEDataset(profile_data=profile_data, traj_len=config["traj_len"])
#     elif config["model_type"] == "fno":
#         dataset = FNODataset(profile_data=profile_data)

#     # Split dataset into training and validation sets
#     dataset_size = len(dataset)
#     val_size = int(config["val_ratio"] * dataset_size)
#     train_size = dataset_size - val_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     # Set up DataLoaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=config["batch_size"], shuffle=True
#     )
#     val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

#     return train_loader, val_loader, profile_data


def setup_data(config):
    data_config = config["data_config"]
    if config["data_type"] == "exp":
        from load_exp_data import ExperimentDataLoader
        from load_exp_data import ExperimentTransformer
        from load_exp_data import ExperimentDataset
        data_loader = ExperimentDataLoader(
            data_dir=data_config["data_dir"],
            experiment_numbers=data_config["exp_nums"],
            valid_solutes=data_config["valid_solutes"],
            valid_substrates=data_config["valid_substrates"],
            valid_temps=data_config["valid_temps"]
        )
        transformer = ExperimentTransformer(
            temporal_subsample=data_config["temporal_subsample"],
            spatial_subsample=data_config["spatial_subsample"],
            axis_symmetric=data_config["axis_symmetric"],
            temporal_pad=data_config["temporal_pad"],
            dtype=torch.float32,
            use_log_transform=data_config["use_log_transform"])

        # Create Dataset
        profile_data = ExperimentDataset(data_loader, transformer)

    elif config["data_type"] == "sim":
        # Load simulation data
        from load_sim_data import SimulationDataLoader
        from load_sim_data import SimulationTransformer
        from load_sim_data import SimulationDataset

        # Load the simulation data
        sim_loader = SimulationDataLoader(data_config["simulation_data_path"])
        sim_data = sim_loader.get_data()

        # Set up the transformer and fit it to the data
        transformer = SimulationTransformer(
            conditioning_keys=data_config.get("conditioning_keys", ["alpha", "beta", "gamma"])
        )
        transformer.fit(sim_data)

        # Create the simulation dataset
        profile_data = SimulationDataset(sim_data, transformer)

    else:
        raise ValueError("Invalid data_type in config. Must be 'experimental' or 'simulation'.")

    if config["model_type"] in ["node", "flux_fno"]:
        dataset = NODEDataset(profile_data=profile_data, traj_len=config["traj_len"])
    elif config["model_type"] == "fno":
        dataset = FNODataset(profile_data=profile_data)

    # Split dataset into training and validation sets
    dataset_size = len(dataset)
    val_size = int(data_config["val_ratio"] * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set up DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=data_config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=data_config["batch_size"], shuffle=False)

    return train_loader, val_loader, dataset
