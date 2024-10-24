import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import random


import read_data


class ODEFunc(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=128, num_hidden_layers=4, activation_fn=nn.ReLU
    ):
        super(ODEFunc, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.model(x)


class NeuralODE(nn.Module):
    def __init__(self, ode_func, solver="rk4"):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.solver = solver

    def forward(self, x0, t):
        return odeint(self.ode_func, x0, t, method=self.solver)


def create_trajectory_data_variable(x, traj_len=32):
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
):
    train_losses = []
    val_losses = []

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
        train_losses.append(avg_train_loss)

        val_loss = validate_model(model, time_steps, val_loader, loss_fn)
        val_losses.append(val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )

    return train_losses, val_losses


def main():
    traj_len = 64
    num_epochs = 4
    lr = 1e-3
    hidden_dim = 64
    num_hidden_layers = 2
    batch_size = 32

    dataset, data_labels = read_data.load_first_sheet_data()

    x_init, y_traj = create_trajectory_data_variable(dataset, traj_len=traj_len)
    train_loader, val_loader, normalizer = read_data.setup_dataloaders(
        x_init, y_traj, test_size=0.1, batch_size=batch_size
    )

    # Initialize ODE model, loss function, and optimizer
    input_dim = dataset.shape[1]
    ode_func = ODEFunc(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        activation_fn=torch.nn.ReLU,
    )
    model = NeuralODE(ode_func)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the Neural ODE model
    train_losses, val_losses = train_node(
        model, traj_len, train_loader, val_loader, optimizer, loss_fn, num_epochs
    )

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
    plt.show()

    for i in range(normed_dataset.shape[1]):
        plt.plot(normed_dataset[:, i], label=data_labels[i])
    plt.xlabel("t")
    plt.legend()
    plt.show()

    x_init_t = 1
    t = torch.linspace(
        0, (len(normed_dataset) - x_init_t)//traj_len, steps=len(normed_dataset) - x_init_t
    )

    with torch.no_grad():
        initial_state = normed_dataset[x_init_t].unsqueeze(0)
        pred_traj = model(initial_state, t)
        x_hist = pred_traj.squeeze()

    for i in range(normed_dataset.shape[1]):
        plt.plot(normed_dataset[:, i], label=f"True: {data_labels[i]}")
        plt.plot(range(x_init_t, len(normed_dataset)), x_hist[:, i], label=f"Predicted: {data_labels[i]}")
    plt.xlabel("t")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
