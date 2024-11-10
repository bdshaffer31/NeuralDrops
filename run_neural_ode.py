import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks
import visualize
import logger
import load_data


def validate_node_model(model, time_steps, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_0, z, y in val_loader:
            y_pred = model(x_0, z, time_steps).transpose(0, 1)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_node(
    model,
    trajectory_len,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    logger,
):
    # 1 integration step per sample (not a great scheme)
    time_steps = torch.linspace(0, 1, steps=trajectory_len)

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for x_0, z, y_traj in train_loader:
            optimizer.zero_grad()
            pred_y_traj = model(x_0, z, time_steps).transpose(0, 1)
            loss = loss_fn(pred_y_traj, y_traj)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        val_loss = validate_node_model(model, time_steps, val_loader, loss_fn)

        logger.log_metrics(
            metrics={"train_mse": avg_train_loss, "val_mse": val_loss}, epoch=epoch
        )
        logger.save_best_model(model, metric=1 / val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )
    logger.save_model(model, file_name="final_model.pth")


def load_node_model_from_logger(log_loader):
    config = log_loader.load_config()

    input_dim = config.get("input_dim")
    output_dim = config.get("output_dim")
    conditioning_dim = config.get("conditioning_dim")
    hidden_dim = config.get("hidden_dim")
    num_hidden_layers = config.get("num_hidden_layers")
    activation_fn = networks.get_activation(config["activation_fn"])
    output_fn = networks.get_activation(config["output_fn"])

    # Load the best model from the logger
    best_model_path = log_loader.get_relpath("best_model.pth")
    # needs to take activ as a string
    ode_func = networks.ODE_FCNN(
        input_dim=input_dim,
        output_dim=output_dim,
        conditioning_dim=conditioning_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_fn=activation_fn,
        output_fn=output_fn,
    )
    model = networks.NeuralODE(ode_func, solver=config["solver"])

    # Load the best validation model
    best_model_path = log_loader.get_relpath("best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def run_training(config, run_dir):
    exp_logger = logger.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    traj_len = config["traj_len"]
    # if we just set up a data parameters sub dict we can pass it as *args
    # and it wouldn't require editing this when we change config setup
    data = load_data.setup_node_data(
        traj_len,
        data_len=config["data_len"],
        batch_size=config["batch_size"],
        exp_nums=config["exp_nums"],
        valid_solutes=config["valid_solutes"],
        valid_substrates=config["valid_substrates"],
        valid_temps=config["valid_temps"],
        temporal_subsample=config["temporal_subsample"],
        spatial_subsample=config["spatial_subsample"],
        use_log_transform=config["use_log_transform"],
        data_dir=config["data_dir"],
        test_split=config["val_ratio"],
    )
    train_loader, val_loader, profile_data = data

    # Initialize ODE model, loss function, and optimizer
    initial_condition, conditioning, target_snapshots = next(iter(train_loader))
    input_dim = initial_condition.shape[1]
    output_dim = target_snapshots.shape[2]
    conditioning_dim = conditioning.shape[1]
    activation_fn = networks.get_activation(config["activation_fn"])
    output_fn = networks.get_activation(config["output_fn"])
    config["input_dim"] = input_dim
    config["output_dim"] = output_dim
    config["conditioning_dim"] = conditioning.shape[1]

    ode_func = networks.ODE_FCNN(
        input_dim=input_dim,
        conditioning_dim=conditioning_dim,
        output_dim=output_dim,
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        activation_fn=activation_fn,
        output_fn=output_fn,
    )
    model = networks.NeuralODE(ode_func, solver=config["solver"])

    exp_logger.log_config(config)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train the Neural ODE model
    train_node(
        model,
        traj_len,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        config["num_epochs"],
        exp_logger,
    )


def main(train=False):
    config = {
        # training params
        "manual_seed": 42,
        "num_epochs": 10,
        "lr": 1e-2,
        # model params
        "hidden_dim": 256,
        "num_hidden_layers": 4,
        "solver": "rk4",
        "activation_fn": "relu",  # "relu",
        "output_fn": "identity",
        # data params
        "data_dir": "data",
        "data_len": 7000,
        "batch_size": 32,
        "exp_nums": None, #[19, 22, 23, 27],  # if None use all, otherwise give a list of ints
        "valid_solutes": ["PVA"],  # if None keep all solutes, otherwise give a list of strings
        "valid_substrates": None,  # if None keep all substrates, otherwise give a list of strings
        "valid_temps": None,  # if None keep all substrates, otherwise give a list of floats
        "temporal_subsample": 30,  # temporal subsampling on profile data
        "spatial_subsample": 4,
        "use_log_transform": False,
        "traj_len": 64,
        "val_ratio": 0.1,
    }
    torch.manual_seed(config["manual_seed"])

    run_dir = "test3"
    if train:
        run_training(config, run_dir)
    visualize.viz_node_results(run_dir)


if __name__ == "__main__":
    # TODO load config from input + defaults
    # split out train and plotting functions
    main(train=True)
