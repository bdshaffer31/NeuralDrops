import torch
import utils
import visualize

def experiment_data_config():
    data_config = {
        "data_dir": "data",
        "batch_size": 32,
        "exp_nums": utils.good_run_numbers()[:10],  # None = use all experiments
        "valid_solutes": None,  # None = keep all solutes
        "valid_substrates": None,  # None = keep all substrates
        "valid_temps": None,  # None = keep all temperatures
        "temporal_subsample": 15,  # Temporal subsampling of profile data
        "spatial_subsample": 5,
        "temporal_pad": 128,
        "axis_symmetric": False,  # split along x axis
        "use_log_transform": False,
        "val_ratio": 0.1,
    }
    return data_config

def simulation_data_config():
    data_config = {
        "data_dir": "data",
        "batch_size": 32,
        "simulation_data_path": "data\simulation_results.pth",
        "conditioning_keys": ["alpha", "beta", "gamma"],
        "val_ratio": 0.1,
    }
    return data_config

def flux_fno_model_config():
    model_config = {
        # "model_type": "flux_fno",
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
        "flux_fno": flux_fno_model_config,
    }
    return model_configs[model_type]()

def get_data_config(data_type):
    data_configs = {
        "exp": experiment_data_config,
        "sim": simulation_data_config,
    }
    return data_configs[data_type]()

def main(train=False):
    # if a config file isn't provided load from options
    run_config = {
        "manual_seed": 42,
        "num_epochs": 10,
        "lr": 1e-2,
        "model_type": "fno", # specify model type
        "data_type": "sim", # specify data type
    }
    run_config["model_config"] = get_model_config(run_config["model_type"])
    run_config["data_config"] = get_data_config(run_config["data_type"])

    torch.manual_seed(run_config["manual_seed"])
    torch.set_default_dtype(torch.float64)


    if run_config["model_type"] == "node":
        from run_neural_ode import run_training
    elif run_config["model_type"] == "fno":
        from run_fno import run_training
    elif run_config["model_type"] == "flux_fno":
        from run_flux_fno import run_training

    run_dir = "test_fno_axis_symmetric"
    if train:
        run_training(run_config, run_dir)
    visualize.viz_results(run_dir)


if __name__ == "__main__":
    main(train=True)