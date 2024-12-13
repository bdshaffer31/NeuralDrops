import torch
import visualize
import run


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
    run_dir = "test_fno_axis_symmetric"
    config = {
        "run_dir": run_dir,
        "manual_seed": 42,
        "num_epochs": 10,
        "lr": 1e-2,
        "model_type": "flux_fno",  # specify model type
        "data_file": "data/simulation_results.pth",  # specify data type
        "batch_size": 32,
        "val_ratio": 0.1,
        "run_keys": [1],  # if None use all
        "conditioning_keys": ["alpha", "beta", "gamma"],
        "profile_scale": 100,  # approx 1 / spacial unit order of magnitude
    }

    config["model_config"] = get_model_config(config["model_type"])

    torch.manual_seed(config["manual_seed"])
    torch.set_default_dtype(torch.float32)

    if train:
        run.run_training(config, run_dir)
    visualize.viz_results(run_dir)


if __name__ == "__main__":
    main(train=False)
