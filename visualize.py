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
    error_std = []
    for run_name in dataset.data:
        height_data = dataset.data[run_name]["profile"]
        pred_traj = trajectory_pred(
            config["model_type"], model, dataset, run_name, t_0_idx=1
        )
        error = (height_data[1:] - pred_traj) ** 2
        error_vals.append(torch.mean(error))
        error_std.append(torch.std(error))
        plot_error_maps(log_loader, dataset, run_name, height_data[1:], pred_traj, title=run_name)
        # plot_drop_flow(log_loader, dataset.data, viz_file, height_data[50], title=r"$\mathbf{v}(t,r,z)$")
        # plot_drop_flow(log_loader, dataset.data, viz_file, pred_traj[49], title=r"$\mathbf{v}_\theta(t,r,z)$")
        # plot_volume_comp_over_time(log_loader, dataset.data, viz_file, height_data[1:], pred_traj, title="")
        # plot_cotact_line_over_time(log_loader, dataset.data, viz_file, height_data[1:], pred_traj, title="")
        # plot_total_flux_over_time(log_loader, dataset, viz_file, height_data, model, title="")


    error_vals = np.array(error_vals)
    error_std = np.array(error_std)
    plt.figure(figsize=(4,3))
    std_clipped = np.minimum(error_std, error_vals)
    x = range(1, len(error_vals) + 1)
    plt.scatter(x, np.sqrt(error_vals), color='k')
    # plt.errorbar(x, np.sqrt(error_vals), yerr=np.sqrt(std_clipped), capsize=5, elinewidth=1, color='k', marker='o', markersize=6, linewidth=0)
    # plt.sca(x, np.sqrt(error_vals), yerr=np.sqrt(std_clipped), capsize=5, elinewidth=1, color='k', marker='o', markersize=6, linewidth=0)
    plt.xticks(x, labels=list(dataset.data.keys()))
    plt.ylabel(r"$RMSE\, [m]$")
    # plt.yscale("log")
    plt.xlabel("Experiment #")
    # plt.title("Mean Squared Error per Dataset")
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
    x = range(1, len(train_losses)+1)
    plt.plot(x, train_losses, label="Train Loss", c="dimgray")
    plt.plot(x, val_losses, label="Val Loss", c="r")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
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





def plot_drop_flow(log_loader, data, viz_file, profile, title=""):

    profile -= 1e-7
    profile = torch.clamp(profile, min=0)  # ensure non-negative height
    # h = drop_utils.symmetrize(h) # may be needed
    profile = drop_utils.drop_polynomial_fit(
        profile, 8
    )  # project height on polynomial basis
    profile = drop_utils.symmetrize(profile)
    profile = torch.clamp(profile, min=0)


    t_lin = data[viz_file]["t_lin"]
    r_lin = data[viz_file]["r_lin"]
    z_lin = data[viz_file]["z_lin"] /5#* 1.3 #/ 5
    dt = t_lin[1] - t_lin[0]
    dr = r_lin[1] - r_lin[0]
    dz = z_lin[1] - z_lin[0]

    drop_params = drop_utils.SimulationParams(
        r_grid=torch.max(r_lin),
        hmax0=torch.max(z_lin),
        Nr=r_lin.shape[0],
        Nz=z_lin.shape[0],
        dr=dr,
        dz=dz,
        # defining these here just creates opportunities to mess them up probably
        rho=1,
        sigma=0.072,
        eta=1e-3,
    )
    # drop_params, dt = drop_utils.load_drop_params_from_data(data)
    def smoothing_fn(x): # reconfigure this into flow post function to be consistent
        return drop_utils.gaussian_blur_1d(x, sigma=10)
    drop_model = pure_drop_model.PureDropModel(drop_params, smoothing_fn=smoothing_fn)

    # fig, axs = plt.subplots(3,1, figsize=(10,3))

    # profile_data = profile_data[:, profile_data.shape[1] // 2 :]
    # pred_traj = pred_traj[:, pred_traj.shape[1] // 2 :]

    # t_lin = data[viz_file]["t_lin"]
    # r_lin = data[viz_file]["r_lin"]
    if r_lin[0] >= 0.0:
        r_lin -= torch.mean(r_lin)

    u_grid = drop_model.calc_u_velocity(profile)
    w_grid = drop_model.calc_w_velocity(profile, u_grid)
    flow_magnitude = torch.sqrt(u_grid**2 + w_grid**2)
    # if log_mag:
    #     flow_magnitude = torch.log(flow_magnitude)
    flow_magnitude = drop_viz.set_nans_in_center(flow_magnitude, 6)
    flow_magnitude = drop_viz.set_nans_in_corners(flow_magnitude, 3)
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
        drop_model.r, profile, color="k", linewidth=2, label="$h(r)$"
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

    plt.xlim([0.0, r_lin[-1]])

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label=r"$||v(r, z)||$")
    # plt.title(r"$v(r, z)$")
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.legend()

    plt.tight_layout()
    log_loader.show(plt)


def plot_volume_comp_over_time(log_loader, data, viz_file, profile_hist, pred_hist, title=""):
    t_lin = data[viz_file]["t_lin"]
    r_lin = data[viz_file]["r_lin"]
    
    data_volume_hist = 2 * torch.pi * torch.sum(torch.abs(profile_hist*r_lin), axis=1) / 2
    pred_volume_hist = 2 * torch.pi * torch.sum(torch.abs(pred_hist*r_lin), axis=1) / 2

    fig, ax1 = plt.subplots(figsize=(5, 4))

    # Plot V(t) and V_\theta(t) on the primary y-axis (left)
    ax1.plot(t_lin[1:], data_volume_hist, label=r"$V(t)$", color="k", zorder=20)
    ax1.plot(t_lin[1:], pred_volume_hist, label=r"$V_\theta(t)$", ls="--", color="r", zorder=30)

    # Customize the primary y-axis
    ax1.set_ylabel(r"$V\, [m^3]$")
    ax1.set_xlabel(r"$t\, [s]$")
    ax1.legend(loc="upper left")

    ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter)

    # Create a secondary y-axis for V(t) - V_\theta(t)
    ax2 = ax1.twinx()
    ax2.plot(t_lin[1:], torch.abs(data_volume_hist - pred_volume_hist), label=r"$|V(t) - V_\theta(t)|$", linestyle="-", color="dimgrey", alpha=0.5, zorder=40)

    # Customize the secondary y-axis
    ax2.set_ylabel(r"$|V(t) - V_\theta(t)|\, [m^3]$")
    ax2.legend(loc="upper right")

    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax2.yaxis.set_major_formatter(formatter)

    # Adjust layout
    fig.tight_layout()

    # Show the plot using log_loader
    log_loader.show(plt)

def plot_cotact_line_over_time(log_loader, data, viz_file, profile_hist, pred_hist, title=""):
    t_lin = data[viz_file]["t_lin"]
    r_lin = data[viz_file]["r_lin"]

    def last_index_above_threshold(tensor, threshold):
        mask = tensor > threshold
        indices = torch.where(mask.any(dim=1), mask.int().argmax(dim=1), -1)
        return indices
    
    data_rc_hist = last_index_above_threshold(profile_hist, 1e-7)
    data_rc_hist = r_lin[data_rc_hist]
    data_rc_hist[data_rc_hist>0] = 0
    data_rc_hist *= -1
    pred_rc_hist = last_index_above_threshold(pred_hist, 1e-7)
    pred_rc_hist = r_lin[pred_rc_hist]
    pred_rc_hist[pred_rc_hist>0] = 0
    pred_rc_hist *= -1

    fig, ax1 = plt.subplots(figsize=(5, 4))

    # Plot V(t) and V_\theta(t) on the primary y-axis (left)
    ax1.plot(t_lin[1:], data_rc_hist, label=r"$r_c(t)$", color="k", zorder=20)
    ax1.plot(t_lin[1:], pred_rc_hist, label=r"$r_{c, \theta}(t)$", ls="--", color="r", zorder=30)

    # Customize the primary y-axis
    ax1.set_ylabel(r"$r_c\, [m]$")
    ax1.set_xlabel(r"$t\, [s]$")
    ax1.legend(loc="upper left")

    ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter)

    # Create a secondary y-axis for V(t) - V_\theta(t)
    ax2 = ax1.twinx()
    ax2.plot(t_lin[1:], torch.abs(data_rc_hist - pred_rc_hist), label=r"$|r_c(t) - r_{c, \theta}(t)|$", linestyle="-", color="dimgrey", alpha=0.5, zorder=40)

    # Customize the secondary y-axis
    ax2.set_ylabel(r"$|r_c(t) - r_{c, \theta}(t)|\, [m]$")
    ax2.legend(loc="upper right")

    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax2.yaxis.set_major_formatter(formatter)

    # Adjust layout
    fig.tight_layout()

    # Show the plot using log_loader
    log_loader.show(plt)


def plot_total_flux_over_time(log_loader, dataset, viz_file, profile_hist, model, title=""):
    # only works with flux-FNO
    t_lin = dataset.data[viz_file]["t_lin"]
    r_lin = dataset.data[viz_file]["r_lin"]

    t_0_idx = 0

    with torch.no_grad():
        conditioning = dataset.get_conditioning(dataset.data[viz_file]).unsqueeze(0)
        evap_model = model.ode_func.model.evap_model #(h0, z)
        scaled_profiles = profile_hist * dataset.profile_scale
        print(conditioning.shape)
        stacked_z = conditioning.repeat(scaled_profiles.shape[0], 1)
        print(stacked_z.shape)
        fluxes = evap_model(scaled_profiles, stacked_z)
        fluxes /= dataset.profile_scale

    plt.figure(figsize=(4,3))
    data_idx = np.linspace(0, len(fluxes) - 1, 4, dtype=int) # dont use last
    plt.plot(r_lin, fluxes[data_idx[0]], label=rf"$f_\theta(h_{{t={data_idx[0]+1}}})$")
    plt.plot(r_lin, fluxes[data_idx[1]], label=rf"$f_\theta(h_{{t={data_idx[1]+1}}})$")
    plt.plot(r_lin, fluxes[data_idx[2]], label=rf"$f_\theta(h_{{t={data_idx[2]+1}}})$")
    plt.xlim(0, r_lin[-1])
    plt.xlabel(r"$r\, [m]$")
    plt.ylabel(r"$\mathcal{E}\, [kg/s]$") # units???
    plt.legend()
    log_loader.show(plt)
    
    total_model_flux = 2 * torch.pi * torch.sum(torch.abs(fluxes*r_lin), axis=1) / 2

    plt.plot(total_model_flux)
    log_loader.show(plt)

    data_volume_hist = 2 * torch.pi * torch.sum(torch.abs(profile_hist*r_lin), axis=1) / 2

    total_model_flux = torch.min(torch.stack([data_volume_hist, total_model_flux]), axis=0)[0]
    print(total_model_flux.shape)

    empirical_mass_flux = -1 * (data_volume_hist[1:]-data_volume_hist[:-1])
    extra_mass_flux = total_model_flux[1:] - empirical_mass_flux

    plt.plot(total_model_flux, label="model flux")
    plt.plot(empirical_mass_flux, label="observed mass flux")
    plt.plot(extra_mass_flux, label="difference")
    plt.legend()
    log_loader.show(plt)
    
    # data_rc_hist = last_index_above_threshold(profile_hist, 1e-4)
    print(torch.min(profile_hist,axis=0))
    mask = profile_hist > 1e-7
    data_rc_hist = torch.where(mask.any(dim=1), mask.int().argmax(dim=1), -1)
    data_rc_hist = r_lin[data_rc_hist]
    data_rc_hist[data_rc_hist>0] = 0
    data_rc_hist *= -1

    # plt.plot(data_rc_hist)
    
    r_lin_as = r_lin[r_lin.shape[0]//2:]
    mass_deposition = torch.zeros_like(r_lin_as)
    sigma = torch.max(r_lin_as)/100  # Standard deviation of the Gaussian kernel

    print(r_lin_as)
    print(data_rc_hist)
    print(r_lin_as.shape)
    print(data_rc_hist.shape)
    print("sigma", sigma)

    extra_mass_flux = torch.abs(extra_mass_flux)

    # Loop through each time step and deposit mass
    for t in range(len(data_rc_hist)-1):
        # Create a Gaussian kernel centered at the interface position r[t]
        gaussian_kernel = torch.exp(-((r_lin_as - data_rc_hist[t]) ** 2) / (2 * sigma ** 2))
        gaussian_kernel /= (np.sqrt(2 * torch.pi) * sigma)  # Normalize the kernel

        # Scale the kernel by the mass added at this time step
        mass_contribution = gaussian_kernel * extra_mass_flux[t]

        # Add the contribution to the mass deposited array
        mass_deposition = mass_deposition + mass_contribution
    mass_deposition = torch.tensor(mass_deposition)
    
    plt.plot(r_lin_as, mass_deposition)
    plt.ylabel("Mass deposited")
    plt.xlabel(r"$r\, [m]$")
    log_loader.show(plt)
