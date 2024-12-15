import torch
import visualize
import run


def flux_fno_model_config():
    model_config = {
        # "model_type": "flux_fno",
        "modes": 16,
        "num_fno_layers": 4,
        "fno_width": 16,
        "num_fc_layers": 4,
        "fc_width": 128,
        "activation_fn": "relu",
        "solver": "euler",
        "gamma": 1e-2,  # constant evap for stability
        "flux_model_scale": 1e-1,  # scaling on evap model for stability
        # "profile_scale": 1e6,  # approx 1 / spacial unit order of magnitude
        # "pixel_resolution": 0.000003, # m / pixel
        # "spatial_sampling": 6, # m / pixel
        # "time_inc": 0.05/12, # time increment and temporal sampling
    }
    return model_config


def fno_node_model_config():
    model_config = {
        # "model_type": "fno_node",
        "modes": 16,
        "num_fno_layers": 4,
        "fno_width": 64,
        "num_fc_layers": 4,
        "fc_width": 256,
        "activation_fn": "relu",
        "solver": "euler",
    }
    return model_config


def fno_model_config():
    model_config = {
        # "model_type": "fno",
        "modes": 16,
        "num_fno_layers": 4,
        "fno_width": 256,
        "num_fc_layers": 4,
        "fc_width": 256,
        "activation_fn": "relu",
    }
    return model_config


def node_model_config():
    model_config = {
        # "model_type": "node",
        "hidden_dim": 256,
        "num_hidden_layers": 6,
        "solver": "rk4",
        "activation_fn": "relu",
        "output_fn": "identity",
    }
    return model_config


def get_model_config(model_type):
    model_configs = {
        "fno": fno_model_config,
        "node": node_model_config,
        "fno_node": fno_node_model_config,
        "flux_fno": flux_fno_model_config,
    }
    return model_configs[model_type]()


def main(train=False):
    # if a config file isn't provided load from options
    run_dir = "fno_flux_deegan_test"
    config = {
        "run_dir": run_dir,
        "manual_seed": 42,
        "num_epochs": 1,
        "traj_len": 8,
        "lr": 1e-2,
        "model_type": "flux_fno",  # specify model type
        "data_file": "data/simulation_results_deegan.pth",  # specify data type
        "batch_size": 16,
        "val_ratio": 0.1,
        "run_keys": [1],  # if None use all
        "conditioning_keys": ["alpha", "beta", "gamma"],
        "profile_scale": 1e3,  # approx 1 / spacial unit order of magnitude
    }

    config["model_config"] = get_model_config(config["model_type"])

    torch.manual_seed(config["manual_seed"])
    torch.set_default_dtype(torch.float64)

    if train:
        run.run_training(config, run_dir)
    visualize.viz_results(run_dir)


if __name__ == "__main__":
    main(train=True)
