import matplotlib.pyplot as plt
import torch
import numpy as np

import logger
import setup_dataloader
import run_neural_ode
import run_fno
import run_flux_fno


def fno_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        time_steps = dataset.data[run_name]["t"][t_0_idx:]
        initial_state = (
            dataset.data[run_name]["profile"][t_0_idx]
            .unsqueeze(0)
            .expand(len(time_steps), -1)
        )
        conditioning = dataset.get_conditioning(run_name)
        conditioning = conditioning.unsqueeze(0).expand(len(time_steps), -1)
        time_steps = time_steps.unsqueeze(1)
        pred_traj = model(initial_state, conditioning, time_steps).squeeze()
        return pred_traj


def node_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        conditioning = dataset.get_conditioning(run_name)
        initial_state = dataset.data[run_name]["profile"][t_0_idx].unsqueeze(0)
        time_steps = dataset.data[run_name]["t"][t_0_idx:]
        print(initial_state.shape, conditioning.shape, time_steps.shape)
        pred_traj = model(initial_state, conditioning, time_steps).squeeze()
        return pred_traj


def flux_fno_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        conditioning = dataset.get_conditioning(run_name).unsqueeze(0)
        initial_state = dataset.data[run_name]["profile"][t_0_idx].unsqueeze(0)
        time_steps = dataset.data[run_name]["t"][t_0_idx:]
        pred_traj = model(initial_state, conditioning, time_steps).squeeze()
        return pred_traj


def trajectory_pred(model_type, model, dataset, run_name, t_0_idx=1):
    if model_type == "node":
        pred_traj = node_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    elif model_type == "fno":
        pred_traj = fno_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    elif model_type == "flux_fno":
        pred_traj = flux_fno_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    return pred_traj


def viz_results(run_dir):
    log_loader = logger.LogLoader(run_dir)
    config = log_loader.load_config()
    metrics = log_loader.load_metrics()
    train_losses = metrics["train_mse"]
    val_losses = metrics["val_mse"]

    dataset = setup_dataloader.setup_data(config)[2]

    if config["model_type"] == "node":
        model = run_neural_ode.load_node_model_from_logger(log_loader)
    elif config["model_type"] == "fno":
        model = run_fno.load_fno_model_from_logger(log_loader)
    elif config["model_type"] == "flux_fno":
        model = run_flux_fno.load_fno_model_from_logger(log_loader)

    # file name for detailed visualization
    viz_file = dataset.valid_files[0]

    plot_train_val_error(log_loader, train_losses, val_losses)
    height_data = dataset.data[viz_file]["profile"]
    plot_profile_stack(log_loader, height_data)

    pred_traj = trajectory_pred(
        config["model_type"], model, dataset, viz_file, t_0_idx=1
    )

    for data_state, pred_state in zip(height_data[::100], pred_traj[::100]):
        plt.plot(data_state, c="dimgrey")
        plt.plot(pred_state, c="r")
    plt.ylabel("h")
    plt.xlabel("X")
    log_loader.show(plt)

    error = (height_data[1:] - pred_traj) ** 2
    error_ts = np.mean(error.numpy(), axis=1)
    plt.plot(error_ts, c="dimgrey")
    plt.ylabel("MSE")
    plt.xlabel("t")
    log_loader.show(plt)

    # loop over all runs in the dataset
    error_vals = []
    for run_name in dataset.data:
        height_data = dataset.data[run_name]["profile"]
        pred_traj = trajectory_pred(
            config["model_type"], model, dataset, run_name, t_0_idx=1
        )
        mse_val = torch.mean((height_data[1:] - pred_traj) ** 2)
        error_vals.append(mse_val)
        plot_error_maps(log_loader, height_data[1:], pred_traj, title=run_name)

    plt.scatter(range(1, len(error_vals) + 1), error_vals)
    plt.xticks(range(1, len(error_vals) + 1), labels=list(dataset.data.keys()))
    plt.ylabel("MSE")
    plt.title("Mean Squared Error per Dataset")
    log_loader.show(plt)


def plot_error_maps(log_loader, dataset, pred_traj, title=""):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))  # 3 rows, 1 column of subplots
    colobar_scale = 0.8
    shape = dataset.shape
    x_ticks = [
        0,
        dataset.shape[1] * 0.25,
        dataset.shape[1] * 0.5,
        dataset.shape[1] * 0.75,
        dataset.shape[1] - 1,
    ]
    x_labels = [0, 0.25, 0.5, 0.75, 1]

    y_ticks = [0, dataset.shape[0] * 0.5, dataset.shape[0] - 1]
    y_labels = [0, 0.5, 1]

    # Plot 1: dataset
    im1 = axs[0].imshow(dataset, aspect="auto", cmap="magma")
    axs[0].invert_yaxis()
    axs[0].set_title("True Drop Height")
    axs[0].set_xticks(x_ticks)
    axs[0].set_xticklabels(x_labels)
    axs[0].set_yticks(y_ticks)
    axs[0].set_yticklabels(y_labels)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("t")
    fig.colorbar(im1, ax=axs[0], shrink=colobar_scale)

    # Plot 2: pred_traj
    im2 = axs[1].imshow(pred_traj, aspect="auto", cmap="magma")
    axs[1].invert_yaxis()
    axs[1].set_title("Model Prediction")
    axs[1].set_xticks(x_ticks)
    axs[1].set_xticklabels(x_labels)
    axs[1].set_yticks(y_ticks)
    axs[1].set_yticklabels(y_labels)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("t")
    fig.colorbar(im2, ax=axs[1], shrink=colobar_scale)

    # Plot 3: Squared Difference
    im3 = axs[2].imshow(np.abs(dataset - pred_traj), aspect="auto", cmap="magma")
    axs[2].invert_yaxis()
    axs[2].set_title("Absolute Difference")
    axs[2].set_xticks(x_ticks)
    axs[2].set_xticklabels(x_labels)
    axs[2].set_yticks(y_ticks)
    axs[2].set_yticklabels(y_labels)
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("t")
    fig.colorbar(im3, ax=axs[2], shrink=colobar_scale)

    # Adjust layout and show/save using logger
    fig.suptitle(title)
    plt.tight_layout()
    log_loader.show(plt)


def plot_train_val_error(log_loader, train_losses, val_losses):
    # Plot the training and validation losses
    plt.plot(train_losses, label="Train Loss", c="dimgray")
    plt.plot(val_losses, label="Val Loss", c="r")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    # plt.xscale('log')
    plt.yscale("log")
    plt.legend()
    log_loader.show(plt)


def plot_profile_stack(log_loader, profiles, title=""):
    for state in profiles[::100]:
        plt.plot(state, c="dimgrey")
    plt.ylabel("h")
    plt.xlabel("X")
    log_loader.show(plt)
