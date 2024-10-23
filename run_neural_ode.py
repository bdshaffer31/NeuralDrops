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
    def __init__(self, input_dim, hidden_dim=128, num_hidden_layers=4, activation_fn=nn.ReLU):
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
    def __init__(self, ode_func, solver='rk4'):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.solver = solver

    def forward(self, x0, t):
        return odeint(self.ode_func, x0, t, method=self.solver)


def norm(data, mean, std):
    return (data - mean) / std

def shuffle_data(x_init, y_traj):
    """ shuffle training data lists """
    combined = list(zip(x_init, y_traj))
    random.shuffle(combined)
    x_init_shuffled, y_traj_shuffled = zip(*combined)
    return list(x_init_shuffled), list(y_traj_shuffled)

def create_trajectory_data_variable(x, max_len=32):
    """
    Create training data where each time step acts as an initial condition,
    and the subsequent points in the trajectory form the ground truth, with variable lengths.
    """
    x_init = []
    y_traj = []

    for i in range(len(x) - 2):
        x_init.append(x[i])
        y_traj.append(x[i+1:i+max_len])
    
    return x_init, y_traj

def train_model_ode_variable_steps(model, x_init, y_traj, optimizer, loss_fn, num_epochs):
    train_losses = []

    total_steps = num_epochs * len(x_init)
    pbar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(num_epochs):
        model.train()

        for i in range(len(x_init)):
            optimizer.zero_grad()

            initial_state = x_init[i].unsqueeze(0)
            true_traj = y_traj[i]

            time_steps = torch.linspace(0, len(true_traj), steps=len(true_traj))
            pred_traj = model(initial_state, time_steps)

            loss = 0.0
            for t_step in range(len(time_steps)):
                loss += loss_fn(pred_traj[t_step,0], true_traj[t_step])

            loss.backward()
            optimizer.step()

            pbar.update(1)

            train_losses.append(loss.item())
            pbar.set_postfix(
                {"Train Loss": f"{loss.item():.4e}"}
            )
    
    return train_losses


def main():
    all_data_dict = read_data.load_drop_data_from_xlsx()
    first_dataset = all_data_dict["0.5wt% 20C"]
    data_inds = [1, 2, -1]
    data_labels = ['Radius', "Height", "Contact Angle"]
    dataset = first_dataset[1:-20, data_inds]

    dataset = norm(dataset, torch.mean(dataset, axis=0), torch.std(dataset, axis=0))

    x_init, y_traj = create_trajectory_data_variable(dataset)
    x_init_shuffled, y_traj_shuffled = shuffle_data(x_init, y_traj)

    # Initialize ODE model, loss function, and optimizer
    input_dim = len(data_inds)
    ode_func = ODEFunc(input_dim=input_dim, hidden_dim=64, num_hidden_layers=2, activation_fn=torch.nn.Tanh)
    model = NeuralODE(ode_func)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the Neural ODE model
    num_epochs = 5
    train_losses = train_model_ode_variable_steps(
        model, x_init_shuffled, y_traj_shuffled, optimizer, loss_fn, num_epochs
    )

    # Plot the training and validation losses
    plt.plot(train_losses, label="Train Loss", c="dimgray")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    # plt.xscale('log')
    plt.yscale("log")
    plt.legend()
    plt.show()

    for i in range(dataset.shape[1]):
        plt.plot(dataset[:,i], label=data_labels[i])
    plt.xlabel('t')
    plt.legend()
    plt.show()

    t = torch.linspace(0, len(dataset)-1, steps=len(dataset)-1)

    with torch.no_grad():
        initial_state = x_init[0].unsqueeze(0)
        pred_traj = model(initial_state, t)
        x_hist = pred_traj.squeeze()

    for i in range(dataset.shape[1]):
        plt.plot(dataset[:,i], label=f'True: {data_labels[i]}')
        plt.plot(t, x_hist[:,i], label=f'Predicted: {data_labels[i]}')
    plt.xlabel('t')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
