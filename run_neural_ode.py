import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks
import visualize
import logger
import utils


def validate_node_model(model, time_steps, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch, time_steps).transpose(0, 1)
            loss = loss_fn(y_pred, y_batch)
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

        for x_0, y_traj in train_loader:
            optimizer.zero_grad()
            pred_y_traj = model(x_0, time_steps).transpose(0, 1)
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
    hidden_dim = config.get("hidden_dim")
    num_hidden_layers = config.get("num_hidden_layers")
    solver = config.get("solver", "rk4")
    activation_fn = networks.get_activation(config["activation_fn"])

    # Load the best model from the logger
    best_model_path = log_loader.get_relpath("best_model.pth")
    # needs to take activ as a string
    ode_func = networks.ODE_FCNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_fn=activation_fn,
    )
    model = networks.NeuralODE(ode_func, solver=solver)

    # Load the best validation model
    best_model_path = log_loader.get_relpath("best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def run_training(config, run_dir):
    logger = logger.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    traj_len = config["traj_len"]
    data = utils.setup_node_data(
        traj_len, config["batch_size"], config["temporal_stride"]
    )
    dataset, train_loader, val_loader, normalizer = data

    # Initialize ODE model, loss function, and optimizer
    # TODO account for conditioning parameters from data
    activation_fn = networks.get_activation(config["activation_fn"])
    input_dim = dataset.shape[1]
    config["input_dim"] = input_dim
    config["output_dim"] = input_dim

    ode_func = networks.ODE_FCNN(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        activation_fn=activation_fn,
    )
    model = networks.NeuralODE(ode_func, solver=config["solver"])

    # TODO log activation
    logger.log_config(config)

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
        logger,
    )


def main(train=False):
    config = {
        "traj_len": 64,
        "num_epochs": 50,
        "lr": 1e-2,
        "hidden_dim": 128,
        "num_hidden_layers": 4,
        "batch_size": 32,
        "solver": "rk4",
        "activation_fn": "relu",
        "manual_seed": 42,
        "temporal_stride": 10,
    }
    torch.manual_seed(config["manual_seed"])

    run_dir = "test"
    if train:
        run_training(config, run_dir)
    visualize.viz_node_results(run_dir)


if __name__ == "__main__":
    # TODO load config from input + defaults
    # split out train and plotting functions
    main(train=False)
