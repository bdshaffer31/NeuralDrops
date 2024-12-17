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
        "gamma": 1e-2,  # constant evap for stability eg 1e-2
        "flux_model_scale": 1e-1,  # scaling on evap model for stability eg 1e-3
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


def main(run_dir):
    # if a config file isn't provided load from options
    
    config = {
        "run_dir": run_dir,
        "manual_seed": 42,
        "num_epochs": 12,
        "traj_len": 8,
        "lr": 1e-3,
        "model_type": "node",  # specify model type ["node", "flux_fno", "fno", "fno_node"]
        "data_file": "data/drop_data_10.pth",  # specify data type "simulation_results_deegan.pth"
        "batch_size": 16,
        "val_ratio": 0.1,
        "run_keys": [1],  # if None use all
        "conditioning_keys": [], #["alpha", "beta", "gamma"],
        "profile_scale": 1e2,  # approx 1 / spacial unit order of magnitude e.g. 1e3
    }

    config["model_config"] = get_model_config(config["model_type"])

    torch.manual_seed(config["manual_seed"])
    torch.set_default_dtype(torch.float64)

    # if train:
    run.run_training(config, run_dir)
    # visualize.viz_results(run_dir)


if __name__ == "__main__":
    # run_dir = "node_drop_data_10"
    # run_dir = "fno_drop_data"
    # run_dir = "fno_flux_deegan"
    run_dir = "fno_flux_drop_data_10"
    # main(run_dir) # comment / uncomment to train or not, yes this is janky ...
    visualize.viz_results(run_dir)
