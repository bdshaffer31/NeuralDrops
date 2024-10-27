import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


import utils


class ODE_FCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        num_hidden_layers=4,
        activation_fn=nn.ReLU,
    ):
        super(ODE_FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.model(x)


# need to learn on log(data) and output exp(model) to prevent negative values
class NeuralODE(nn.Module):
    def __init__(self, ode_func, solver="rk4"):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.solver = solver

    def forward(self, x0, t):
        return odeint(self.ode_func, x0, t, method=self.solver)


def create_trajectories(x, traj_len=32):
    """
    Create training data where each time step acts as an initial condition,
    and the subsequent points in the trajectory form the ground truth, with variable lengths.
    """
    x_init = []
    y_traj = []

    for i in range(len(x) - traj_len):
        x_init.append(x[i])
        y_traj.append(x[i + 1 : i + traj_len + 1])

    x_init = torch.stack(x_init)
    y_traj = torch.stack(y_traj)

    return x_init, y_traj


def validate_model(model, time_steps, val_loader, loss_fn):
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

        val_loss = validate_model(model, time_steps, val_loader, loss_fn)

        logger.log_metrics(
            metrics={"train_mse": avg_train_loss, "val_mse": val_loss}, epoch=epoch
        )
        logger.save_best_model(model, metric=1 / val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )
    logger.save_model(model, file_name="final_model.pth")


def load_model_from_logger(log_loader):
    config = log_loader.load_config()

    input_dim = config.get("input_dim")
    output_dim = config.get("output_dim")
    hidden_dim = config.get("hidden_dim")
    num_hidden_layers = config.get("num_hidden_layers")
    solver = config.get("solver", "rk4")
    activation_fn = utils.get_activation(config["activation_fn"])

    # Load the best model from the logger
    best_model_path = log_loader.get_relpath("best_model.pth")
    # needs to take activ as a string
    ode_func = ODE_FCNN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_fn=activation_fn,
    )
    model = NeuralODE(ode_func, solver=solver)

    # Load the best validation model
    best_model_path = log_loader.get_relpath("best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def setup_data(traj_len, batch_size, temporal_stride=10):
    dataset = utils.load_test_mat_file()[::temporal_stride]
    dataset, _ = utils.detrend_dataset(dataset, window_size=150)
    dataset = utils.center_data(dataset)

    traj_len = traj_len
    x_init, y_traj = create_trajectories(dataset, traj_len=traj_len)
    train_loader, val_loader, normalizer = utils.setup_dataloaders(
        x_init, y_traj, test_size=0.1, batch_size=batch_size, feature_norm=False
    )
    return dataset, train_loader, val_loader, normalizer


def run_training(config, run_dir):
    logger = utils.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    traj_len = config["traj_len"]
    data = setup_data(traj_len, config["batch_size"], config["temporal_stride"])
    dataset, train_loader, val_loader, normalizer = data

    # Initialize ODE model, loss function, and optimizer
    # TODO account for conditioning parameters from data
    activation_fn = utils.get_activation(config["activation_fn"])
    input_dim = dataset.shape[1]
    config["input_dim"] = input_dim
    config["output_dim"] = input_dim

    ode_func = ODE_FCNN(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
        activation_fn=activation_fn,
    )
    model = NeuralODE(ode_func, solver=config["solver"])

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


def viz_results(run_dir):
    log_loader = utils.LogLoader(run_dir)
    config = log_loader.load_config()
    metrics = log_loader.load_metrics()
    model = load_model_from_logger(log_loader)
    train_losses = metrics["train_mse"]
    val_losses = metrics["val_mse"]

    traj_len = config["traj_len"]
    data = setup_data(traj_len, config["batch_size"], config["temporal_stride"])
    dataset, train_loader, val_loader, normalizer = data

    normed_dataset = normalizer.apply(dataset)

    # Plot the training and validation losses
    plt.plot(train_losses, label="Train Loss", c="dimgray")
    plt.plot(val_losses, label="Val Loss", c="r")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    # plt.xscale('log')
    plt.yscale("log")
    plt.legend()
    log_loader.show(plt)

    for state in normed_dataset[::100]:
        plt.plot(state, c="dimgrey")
    plt.ylabel("h")
    plt.xlabel("X")
    plt.title("Sample Images")
    log_loader.show(plt)

    x_init_t = 1
    t = torch.linspace(
        0,
        (len(normed_dataset) - x_init_t) // traj_len,
        steps=len(normed_dataset) - x_init_t,
    )

    with torch.no_grad():
        initial_state = normed_dataset[x_init_t].unsqueeze(0)
        pred_traj = model(initial_state, t)
        x_hist = pred_traj.squeeze()

    for data_state, pred_state in zip(normed_dataset[::100], x_hist[::100]):
        plt.plot(data_state, c="dimgrey")
        plt.plot(pred_state, c="r")
    plt.ylabel("h")
    plt.xlabel("X")
    plt.title("Sample Data and Model Predictions from Initial State")
    log_loader.show(plt)

    error = (normed_dataset[1:] - x_hist) ** 2
    error_ts = np.mean(error.numpy(), axis=1)
    plt.plot(error_ts, c="dimgrey")
    plt.ylabel("MSE")
    plt.xlabel("t")
    log_loader.show(plt)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6))  # 3 rows, 1 column of subplots
    colobar_scale = 0.8
    shape = normed_dataset.shape
    x_ticks = [
        0,
        normed_dataset.shape[1] * 0.25,
        normed_dataset.shape[1] * 0.5,
        normed_dataset.shape[1] * 0.75,
        normed_dataset.shape[1] - 1,
    ]
    x_labels = [0, 0.25, 0.5, 0.75, 1]

    y_ticks = [0, normed_dataset.shape[0] * 0.5, normed_dataset.shape[0] - 1]
    y_labels = [0, 0.5, 1]
    # Plot 1: normed_dataset
    im1 = axs[0].imshow(normed_dataset, aspect="equal", cmap="magma")
    axs[0].set_title("Normalized Data")
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("t")
    fig.colorbar(im1, ax=axs[0], shrink=colobar_scale)

    # Plot 2: x_hist
    im2 = axs[1].imshow(x_hist, aspect="equal", cmap="magma")
    axs[1].set_title("Model Prediction from Initial State")
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("t")
    fig.colorbar(im2, ax=axs[1], shrink=colobar_scale)

    # Plot 3: Squared Difference
    im3 = axs[2].imshow(
        np.abs(normed_dataset[1:] - x_hist), aspect="equal", cmap="magma"
    )
    axs[2].set_title("Absolute Difference")
    axs[2].set_xticks(x_ticks)
    axs[2].set_xticklabels(x_labels)
    axs[2].set_yticks(y_ticks)
    axs[2].set_yticklabels(y_labels)
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("t")
    fig.colorbar(im3, ax=axs[2], shrink=colobar_scale)

    # Adjust layout and show/save using logger
    plt.tight_layout()
    log_loader.show(plt)


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
    viz_results(run_dir)


if __name__ == "__main__":
    # TODO load config from input + defaults
    # split out train and plotting functions
    main(train=False)
