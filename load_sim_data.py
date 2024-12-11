import torch
from torch.utils.data import Dataset
from load_exp_data import Standardizer

class SimulationDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        return torch.load(self.file_path)

    def get_data(self):
        return self.data

class SimulationTransformer:
    def __init__(self, conditioning_keys=["alpha", "beta", "gamma"]):
        self.conditioning_keys = conditioning_keys
        self.profile_standardizer = None
        self.conditioning_standardizers = {}

    def fit(self, data):
        """Fit standardizers for profiles and conditioning variables."""
        # Collect all profiles and conditioning values
        profiles = []
        conditioning_values = {key: [] for key in self.conditioning_keys}

        for sample in data.values():
            profiles.append(torch.tensor(sample["profile"], dtype=torch.float32))
            for key in self.conditioning_keys:
                conditioning_values[key].append(sample[key])

        # Fit the profile standardizer
        all_profiles = torch.cat(profiles, dim=0)
        self.profile_standardizer = Standardizer(all_profiles)

        # Fit standardizers for each conditioning key
        for key in self.conditioning_keys:
            values = torch.tensor(conditioning_values[key], dtype=torch.float32)
            self.conditioning_standardizers[key] = Standardizer(values)

    def transform(self, sample):
        """Standardize the profile and conditioning variables in a sample."""
        profile = torch.tensor(sample["profile"], dtype=torch.float32)
        standardized_profile = self.profile_standardizer.apply(profile)

        standardized_conditioning = torch.tensor(
            [self.conditioning_standardizers[key].apply(torch.tensor(sample[key])) for key in self.conditioning_keys],
            dtype=torch.float32
        )

        return standardized_profile, standardized_conditioning

    def inverse_transform_profile(self, profile):
        """Unstandardize a profile."""
        return self.profile_standardizer.inverse_apply(profile)

    def inverse_transform_conditioning(self, conditioning):
        """Unstandardize conditioning values."""
        unstandardized = [
            self.conditioning_standardizers[key].inverse_apply(cond) 
            for key, cond in zip(self.conditioning_keys, conditioning)
        ]
        return torch.tensor(unstandardized, dtype=torch.float32)


class SimulationDataset(Dataset):
    def __init__(self, data, transformer, conditioning_keys=["alpha", "beta", "gamma"]):
        """
        Args:
            data (dict): Dictionary of simulation results.
            transformer (SimulationTransformer): Transformer for standardization.
        """
        self.data = data
        self.transformer = transformer
        self.conditioning_keys = conditioning_keys
        self.keys = list(data.keys())  # To access each entry by index
    
    def get_conditioning(self, sample):
        conditioning = torch.tensor(
            [self.data[sample][key] for key in self.conditioning_keys],
            # dtype=self.transformer.dtype,
        )
        return conditioning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]
        standardized_profile, standardized_conditioning = self.transformer.transform(sample)
        return standardized_profile, standardized_conditioning


def main():
    import matplotlib.pyplot as plt
    # Load the data using the SimulationDataLoader
    loader = SimulationDataLoader("data/simulation_results.pth")
    data = loader.get_data()

    # Initialize and fit the transformer
    transformer = SimulationTransformer(conditioning_keys=["alpha", "beta", "gamma"])
    transformer.fit(data)

    # Create the SimulationDataset
    dataset = SimulationDataset(data, transformer)

    # Example: Iterate through the dataset
    for profile, conditioning in dataset:
        print("Standardized Profile shape:", profile.shape)
        print("Standardized Conditioning values:", conditioning)

    # Example: Unstandardize a profile and conditioning
    profile, conditioning = dataset[0]
    unstandardized_profile = transformer.inverse_transform_profile(profile)
    unstandardized_conditioning = transformer.inverse_transform_conditioning(conditioning)

    print("Unstandardized Profile:", unstandardized_profile)
    print("Unstandardized Conditioning:", unstandardized_conditioning)

    # Example: Iterate through the dataset
    for profile_hist, conditioning in dataset:
        print("Profile shape:", profile_hist.shape)
        print("Conditioning values:", conditioning)
        plt.plot(profile_hist[0])
        plt.plot(profile_hist[-1])
        plt.show()


if __name__ == "__main__":
    main()
