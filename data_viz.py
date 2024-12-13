import torch
import numpy as np
from scipy.io import loadmat
import os
import logger
import matplotlib.pyplot as plt


import utils_old


def load_contents(data_dir, file):
    contents = loadmat(os.path.join(data_dir, file))
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


# def load_profile_from_contents(
#     contents_data, end_pad=64, temporal_subsample=10, spatial_subsample=1
# ):
#     np_profile = contents_data["profile"]
#     np_profile = np.array(np_profile, dtype="float32")
#     profile = torch.tensor(np_profile, dtype=torch.float32)
#     # apply detrending, centering, padding here
#     profile, _ = utils.detrend_dataset(profile, last_n=50, window_size=50)
#     profile = utils.center_data(profile)
#     profile = utils.pad_profile(profile, end_pad * temporal_subsample)
#     profile = utils.smooth_profile(profile)
#     profile = utils.vertical_crop_profile(profile, 0.78)

#     profile = profile[::temporal_subsample, ::spatial_subsample]
#     return profile


def load_profile_from_contents(
    contents_data, end_pad=64, temporal_subsample=10, spatial_subsample=1
):
    # put all preprocessing to load profile data here
    # leaving as individual
    np_profile = contents_data["profile"]
    np_profile = np.array(np_profile, dtype="float32")
    profile = torch.tensor(np_profile, dtype=torch.float32)
    # apply detrending, centering, padding here
    profile, _ = utils_old.detrend_dataset(profile, last_n=50, window_size=50)
    profile = utils_old.smooth_profile(profile)
    profile = utils_old.pad_profile(profile, end_pad * temporal_subsample)
    profile = utils_old.vertical_crop_profile(profile, 0.78)
    profile = utils_old.center_data(profile)
    profile = utils_old.central_split_data(profile)

    profile = profile[::temporal_subsample, ::spatial_subsample]
    return profile


def load_profile(data_dir, file):
    contents = load_contents(data_dir, file)
    return load_profile_from_contents(contents)


def plot_profile_stacked(profile, exp_logger, step=100, title=""):
    for state in profile[::step]:
        plt.plot(state, c="dimgrey")
    plt.ylabel("h")
    plt.xlabel("X")
    plt.title(title)
    exp_logger.show(plt)


def plot_profile_imshow(profile, exp_logger, title=""):
    plt.imshow(profile, cmap="magma", aspect="auto")
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title(title)
    exp_logger.show(plt)


def plot_all_profiles_mean(all_profiles):
    pass


if __name__ == "__main__":
    # contents = loadmat(os.path.join("data", "Exp_31.mat"))
    # breakpoint()

    data_dir = "data"
    files = [
        filename
        for filename in os.listdir(data_dir)
        if filename.startswith("Exp_") and filename.endswith(".mat")
    ]
    # for file in files:
    #     contents = load_contents(data_dir, file)
    #     print(file, contents["profile"].shape)

    exp_logger = logger.ExperimentLogger(run_dir="data_test_viz2", use_timestamp=False)
    for file in files:
        print(file)
        profile = load_profile(data_dir, file)
        plot_profile_stacked(profile, exp_logger, title=file)
        plot_profile_imshow(profile, exp_logger, title=file)
