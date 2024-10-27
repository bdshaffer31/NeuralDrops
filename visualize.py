import matplotlib.pyplot as plt
import torch
import numpy as np

import logger
import utils
import run_neural_ode


def viz_node_results(run_dir):
    log_loader = logger.LogLoader(run_dir)
    config = log_loader.load_config()
    metrics = log_loader.load_metrics()
    model = run_neural_ode.load_node_model_from_logger(log_loader)
    train_losses = metrics["train_mse"]
    val_losses = metrics["val_mse"]

    traj_len = config["traj_len"]
    data = utils.setup_node_data(
        traj_len, config["batch_size"], config["temporal_stride"]
    )
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
