import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import numpy as np

import logger
import load_data
import run
import matplotlib.ticker as ticker

from drop_model import drop_viz
import drop_model.utils as drop_utils
from drop_model import pure_drop_model


def viz_results(run_dir):
    drop_viz.set_styling()
    log_loader = logger.LogLoader(run_dir)
    config = log_loader.load_config()
    metrics = log_loader.load_metrics()
    train_losses = metrics["train_mse"]
    val_losses = metrics["val_mse"]

    dataset = load_data.setup_data_from_config(config)[2]

    model = run.load_model_from_log_loader(log_loader)

    viz_file = list(dataset.data.keys())[0]

    detailed_viz(model, dataset, viz_file)

    plot_train_val_error(log_loader, train_losses, val_losses)
    height_data = dataset.data[viz_file]["profile"]
    plot_profile_stack(log_loader, height_data)

    pred_traj = trajectory_pred(
        config["model_type"], model, dataset, viz_file, t_0_idx=1
    )

    plt.plot(height_data[0], c="k", linewidth=2)
    plt.plot(pred_traj[0], c="maroon", linewidth=2)
    log_loader.show(plt)

    plt.plot(height_data[0], c="k", linewidth=2)
    plt.plot(pred_traj[0], c="maroon", linewidth=2)
    for data_state, pred_state in zip(height_data[::100], pred_traj[::100]):
        plt.plot(data_state, c="dimgrey")
        plt.plot(pred_state, c="r")
    plt.plot(height_data[-1], c="k", linewidth=2)
    plt.plot(pred_traj[-1], c="maroon", linewidth=2)
    plt.ylabel("h")
    plt.xlabel("X")
    log_loader.show(plt)

    error = (height_data[1:] - pred_traj) ** 2
    error_ts = np.mean(error.numpy(), axis=1)
    plt.plot(error_ts, c="dimgrey")
    plt.ylabel("MSE")
    plt.xlabel("t")
    log_loader.show(plt)

    rmse = torch.sqrt(error)
    data_rmse = dataset.data[viz_file]['profile'] * dataset.profile_scale
    print(f"data rmse: {torch.mean(data_rmse):.4e}")
    print(f"mean rmse: {torch.mean(rmse):.4e}")
    print(f"std rmse: {torch.std(rmse):.4e}")

    # loop over all runs in the dataset
    error_vals = []
    for run_name in dataset.data:
        height_data = dataset.data[run_name]["profile"]
        pred_traj = trajectory_pred(
            config["model_type"], model, dataset, run_name, t_0_idx=1
        )
        mse_val = torch.mean((height_data[1:] - pred_traj) ** 2)
        error_vals.append(mse_val)
        plot_error_maps(log_loader, dataset, run_name, height_data[1:], pred_traj, title=run_name)
        # plot_drop_flow()..... for h and h_\theta

    plt.scatter(range(1, len(error_vals) + 1), error_vals)
    plt.xticks(range(1, len(error_vals) + 1), labels=list(dataset.data.keys()))
    plt.ylabel("MSE")
    plt.title("Mean Squared Error per Dataset")
    log_loader.show(plt)


def fno_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        time_steps = dataset.data[run_name]["t_lin"][t_0_idx:]
        initial_state = (
            dataset.data[run_name]["profile"][t_0_idx]
            .unsqueeze(0)
            .expand(len(time_steps), -1)
        )
        initial_state = initial_state * dataset.profile_scale
        conditioning = dataset.get_conditioning(dataset.data[run_name])
        conditioning = conditioning.unsqueeze(0).expand(len(time_steps), -1)
        time_steps = time_steps.unsqueeze(1)
        pred_traj = model(initial_state, conditioning, time_steps).squeeze()
        return pred_traj


def node_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        conditioning = dataset.get_conditioning(dataset.data[run_name])
        initial_state = dataset.data[run_name]["profile"][t_0_idx].unsqueeze(0)
        initial_state = initial_state * dataset.profile_scale
        time_steps = dataset.data[run_name]["t_lin"][t_0_idx:]
        pred_traj = model(initial_state, conditioning, time_steps).squeeze()
        return pred_traj


def fno_node_traj_pred_from_dataset(model, dataset, run_name, t_0_idx=1):
    with torch.no_grad():
        conditioning = dataset.get_conditioning(dataset.data[run_name]).unsqueeze(0)
        initial_state = dataset.data[run_name]["profile"][t_0_idx].unsqueeze(0)
        initial_state = initial_state * dataset.profile_scale
        time_steps = dataset.data[run_name]["t_lin"][t_0_idx:]
        time_steps = torch.arange(time_steps.shape[0]).to(torch.float64)

        sub_step = 9
        total_points = (len(time_steps) - 1) * (sub_step + 1) + 1
        t_sub_stepped = torch.linspace(
            time_steps[0], time_steps[-1], steps=total_points
        )

        pred_traj = model(initial_state, conditioning, t_sub_stepped).squeeze()
        pred_y_traj_original_t = pred_traj[:: sub_step + 1]
        return pred_y_traj_original_t


def trajectory_pred(model_type, model, dataset, run_name, t_0_idx=1):
    if model_type == "node":
        pred_traj = node_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    elif model_type == "fno":
        pred_traj = fno_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    elif model_type in ["fno_node", "flux_fno"]:
        pred_traj = fno_node_traj_pred_from_dataset(
            model, dataset, run_name, t_0_idx=t_0_idx
        )
    return pred_traj / dataset.profile_scale # rescale to meters


def detailed_viz(dataset, model, viz_file):
    pass


def plot_error_maps(log_loader, dataset, viz_file, profile_data, pred_traj, title=""):
    t_lin = dataset.data[viz_file]["t_lin"]
    r_lin = dataset.data[viz_file]["r_lin"]
    if r_lin[0] >= 0.0:
        r_lin -= torch.mean(r_lin)
    t_indices = np.linspace(0, len(t_lin) - 1, 3, dtype=int)
    r_indices = np.linspace(0, len(r_lin) - 1, 5, dtype=int)

    t_ticks = t_lin[t_indices].numpy()
    r_ticks = r_lin[r_indices]
    r_ticks[0] = 0.0


    fig, axs = plt.subplots(figsize=(10, 3), frameon=True)  # 3 rows, 1 column of subplots
    plt.gca().axis('off')
    fig.add_subplot(111, frameon=False)  # Add a frame-less subplot covering the whole figure
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    gs = GridSpec(1, 6, width_ratios=[1, 1, 0.05, 0.05, 1, 0.05], wspace=0.4)
    colobar_scale = 0.9
    # shape = profile_data.shape
    # x_ticks = [
    #     0,
    #     profile_data.shape[1] * 0.25,
    #     profile_data.shape[1] * 0.5,
    #     profile_data.shape[1] * 0.75,
    #     profile_data.shape[1] - 1,
    # ]
    # x_labels = [0, 0.25, 0.5, 0.75, 1]

    # y_ticks = [0, profile_data.shape[0] * 0.5, profile_data.shape[0] - 1]
    # y_labels = [0, 0.5, 1]

    stacked_data = torch.stack([torch.zeros_like(profile_data), profile_data, pred_traj])
    vmin = max(0.0, torch.min(stacked_data))
    vmax = torch.max(stacked_data)

    def add_colorbar(im, ax, vmin, vmax, label=False):
        cbar = fig.colorbar(im, ax=ax, shrink=colobar_scale)
        # Set 5 tick values evenly spaced between data_vmin and data_vmax
        tick_values = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(tick_values)
        if label:
            cbar.set_label("[m]")
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        # formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2g}")
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.ax.yaxis.get_offset_text().set_fontsize(10)
        cbar.ax.yaxis.get_offset_text().set_x(2.0)
    
    def set_scientific_ticks(ax):
        pass
        # Formatter for the x-axis
        # x_formatter = ticker.ScalarFormatter(useMathText=True)
        # x_formatter.set_scientific(True)
        # x_formatter.set_powerlimits((-2, 2))
        # ax.xaxis.set_major_formatter(x_formatter)
        
        # Formatter for the y-axis
        # y_formatter = ticker.ScalarFormatter(useMathText=True)
        # y_formatter.set_scientific(True)
        # y_formatter.set_powerlimits((-2, 2))
        # ax.yaxis.set_major_formatter(y_formatter)

    # Plot 1: dataset
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(profile_data, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,)
    ax1.invert_yaxis()
    ax1.set_title(r"$h(t)$") #, \, [m]
    ax1.set_xticks(r_indices)
    ax1.set_xticklabels([f"{r:.2e}" for r in r_ticks])
    ax1.set_yticks(t_indices)
    ax1.set_yticklabels([f"{t:.2f}" for t in t_ticks])
    set_scientific_ticks(ax1)
    ax1.set_xlabel(r"$r\, [m]$")
    ax1.set_ylabel(r"$t\, [s]$")
    ax1.set_xlim(left=profile_data.shape[1]/2)
    ax1.grid(False)
    # add_colorbar(im1, ax1, vmin, vmax)

    # Plot 2: pred_traj
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_traj, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,)
    ax2.invert_yaxis()
    ax2.set_title(r"$h_\theta(t) = \int_{t_0}^{t} \mathcal{G} - f_\theta(h(\tau), z)\, d\tau$") #, \, [m]
    ax2.set_xticks(r_indices)
    ax2.set_xticklabels([f"{r:.2e}" for r in r_ticks])
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    set_scientific_ticks(ax2)
    ax2.set_xlabel(r"$r\, [m]$")
    ax2.set_xlim(left=profile_data.shape[1]/2)
    ax2.grid(False)
    # add_colorbar(im2, ax2, vmin, vmax)

    cbar_ax = fig.add_subplot(gs[0, 2])  # Colorbar in the third column
    cbar = fig.colorbar(im2, cax=cbar_ax, shrink=colobar_scale)
    cbar.set_label("[m]")
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter)
    # fig.add_subplot(gs[0, 3])  # Colorbar in the third column

    # Plot 3: Squared Difference
    ax3 = fig.add_subplot(gs[0, 4])
    im3 = ax3.imshow(np.abs(profile_data - pred_traj), aspect="auto", cmap="magma",)
    ax3.invert_yaxis()
    ax3.set_title(r"$||h_\theta(t) - h(t)||_1$") #, \, [m]
    ax3.set_xticks(r_indices)
    ax3.set_xticklabels([f"{r:.2e}" for r in r_ticks])
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    set_scientific_ticks(ax3)
    ax3.set_xlabel(r"$r\, [m]$")
    ax3.set_xlim(left=profile_data.shape[1]/2)
    ax3.grid(False)

    cbar_ax2 = fig.add_subplot(gs[0, 5])  # Colorbar in the third column
    cbar = fig.colorbar(im3, cax=cbar_ax2, shrink=colobar_scale)
    cbar.set_label("[m]")
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter)
    # add_colorbar(im3, ax3, im3.get_array().min(), im3.get_array().max(), True)


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





def plot_drop_flow(log_loader, dataset, viz_file, profile_data, pred_traj, title=""):
    drop_params, dt = drop_utils.load_drop_params_from_data(dataset.data)
    def smoothing_fn(x): # reconfigure this into flow post function to be consistent
        return drop_utils.gaussian_blur_1d(x, sigma=10)
    drop_model = pure_drop_model.PureDropModel(drop_params, smoothing_fn=smoothing_fn)

    # fig, axs = plt.subplots(3,1, figsize=(10,3))

    # profile_data = profile_data[:, profile_data.shape[1] // 2 :]
    # pred_traj = pred_traj[:, pred_traj.shape[1] // 2 :]

    t_lin = dataset.data[viz_file]["t_lin"]
    r_lin = dataset.data[viz_file]["r_lin"]
    if r_lin[0] >= 0.0:
        r_lin -= torch.mean(r_lin)
    
    h = profile_data[10]

    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)
    flow_magnitude = torch.sqrt(u_grid**2 + w_grid**2)
    # if log_mag:
    #     flow_magnitude = torch.log(flow_magnitude)
    # flow_magnitude = set_nans_in_center(flow_magnitude, center_mask)
    # flow_magnitude = set_nans_in_corners(flow_magnitude, corner_mask)
    flow_magnitude[flow_magnitude == 0.0] = torch.nan

    # plt.figure(figsize=(6, 3))
    # ax2_1 = fig.add_subplot(gs[1, 0])  # Colorbar in the third column
    im = plt.imshow(
        flow_magnitude.T,
        aspect="auto",
        origin="lower",
        extent=[
            -drop_model.params.r_grid,
            drop_model.params.r_grid,
            0,
            torch.max(drop_model.z),
        ],
        cmap="magma",
    )
    plt.plot(
        drop_model.r, h, color="k", linewidth=2, label="$h(r)$"
    )  # Overlay height profile

    # Create a meshgrid for quiver plot
    R, Z = torch.meshgrid(drop_model.r, drop_model.z, indexing="ij")

    # Downsample for quiver plot (to avoid too many arrows)
    quiver_step = 6  # Adjust this for clarity
    R_down = R[::quiver_step, ::quiver_step]
    Z_down = Z[::quiver_step, ::quiver_step]
    U_down = u_grid[::quiver_step, ::quiver_step]
    W_down = w_grid[::quiver_step, ::quiver_step]

    magnitude = torch.sqrt(U_down**2 + W_down**2) + 1e-6  # Avoid division by zero
    U_down_unit = U_down / magnitude
    W_down_unit = W_down / magnitude

    # Plot flow direction using quiver
    plt.quiver(
        R_down,
        Z_down,
        U_down_unit,
        W_down_unit,
        color="white",
        scale=50,
        width=0.003,
        headwidth=3,  # scale=100_000
    )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(
        (-0, 0)
    )  # Force scientific notation for values between 10^-1 and 10^1

    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Ensure the ticks are redrawn
    plt.gca().ticklabel_format(style="sci", axis="both", scilimits=(-0, 0))
    plt.gca().autoscale_view()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label=r"$||v(r, z)||$")
    plt.title(r"$v(r, z)$")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()

    plt.tight_layout()
    log_loader.show(plt)

