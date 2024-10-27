import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils


class FCNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_hidden_layers=4,
        hidden_dim=128,
        activation_fn=torch.nn.ReLU,
    ):
        super(FCNN, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn())
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def norm(data, mean, std):
    return (data - mean) / std


def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=500):
    train_losses = []
    val_losses = []

    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validate the model
        val_loss = validate_model(model, val_loader, loss_fn)
        val_losses.append(val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )

    return train_losses, val_losses


def validate_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def main():
    all_data_dict = utils.load_drop_data_from_xlsx()
    first_dataset = all_data_dict["0.5wt% 20C"]
    data_inds = [1, 2, -1]
    # data_inds = [1, 2]
    data_labels = ["Radius", "Height", "Contact Angle"]
    dataset = first_dataset[1:-20, data_inds]
    x = dataset[:-1]
    y = dataset[1:]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Normalize based on training data
    X_train_mean = torch.mean(X_train, axis=0)
    X_train_std = torch.std(X_train, axis=0)

    y_train_mean = torch.mean(y_train, axis=0)
    y_train_std = torch.std(y_train, axis=0)

    X_train, X_val, x = [
        norm(data, X_train_mean, X_train_std) for data in [X_train, X_val, x]
    ]
    y_train, y_val, y = [
        norm(data, y_train_mean, y_train_std) for data in [y_train, y_val, y]
    ]

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # Initialize model, loss function, and optimizer
    input_dim = len(data_inds)
    output_dim = len(data_inds)
    model = FCNN(
        input_dim,
        output_dim,
        num_hidden_layers=2,
        hidden_dim=64,
        activation_fn=torch.nn.Tanh,
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 100
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, loss_fn, num_epochs
    )

    # Plot the training and validation losses
    plt.plot(train_losses, label="Train Loss", c="dimgray")
    plt.plot(val_losses, label="Validation Loss", c="r")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    # plt.xscale('log')
    plt.yscale("log")
    plt.legend()
    plt.show()

    for i in range(x.shape[1]):
        plt.plot(x[:, i], label=data_labels[i])
    plt.xlabel("t")
    plt.legend()
    plt.show()

    with torch.no_grad():
        current_x = x[0]
        x_hist = [current_x]
        for i in range(len(x)):
            current_x = model(current_x)
            x_hist.append(current_x)
        x_hist = torch.stack(x_hist)

    for i in range(x.shape[1]):
        plt.plot(x[:, i], label=f"True: {data_labels[i]}")
        plt.plot(x_hist[:, i], label=f"Predicted: {data_labels[i]}")
    plt.xlabel("t")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
