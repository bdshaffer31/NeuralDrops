# import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_styling():
    plt.rcParams.update(
        {
            # Figure size and DPI
            "figure.figsize": (6, 4),  # Default figure size (width, height) in inches
            "figure.dpi": 150,  # High-resolution figures for better quality
            # Font settings
            "font.family": "serif",  # Use serif fonts for a professional look
            "font.size": 12,  # Default font size
            "axes.titlesize": 14,  # Title font size
            "axes.labelsize": 12,  # Axis label font size
            "legend.fontsize": 11,  # Legend font size
            "xtick.labelsize": 11,  # X-tick label font size
            "ytick.labelsize": 11,  # Y-tick label font size
            # Line settings
            "lines.linewidth": 1.5,  # Line width for plots
            "lines.markersize": 6,  # Marker size for scatter plots and line markers
            # Grid and ticks
            "axes.grid": True,  # Enable grid
            "grid.linestyle": "--",  # Dashed lines for grid
            "grid.alpha": 0.5,  # Grid transparency
            "xtick.direction": "in",  # Ticks pointing inward
            "ytick.direction": "in",
            # Figure borders (spines)
            "axes.spines.top": True,  # Display top spine
            "axes.spines.right": True,  # Display right spine
            "axes.spines.left": True,  # Display left spine
            "axes.spines.bottom": True,  # Display bottom spine
            # Legends
            "legend.frameon": True,  # Add a frame around legends
            "legend.loc": "best",  # Best location for the legend
            # Save settings
            "savefig.dpi": 300,  # DPI for saved figures
            "savefig.format": "png",  # Default format for saving figures
            # Color map for consistency
            "image.cmap": "plasma",  # Default color map for images
            # Latex for math text (optional, requires LaTeX installed)
            "text.usetex": False,  # Set to True if you want to use LaTeX for text rendering
            "mathtext.fontset": "stix",  # Use STIX fonts for math text
        }
    )


def grad(x, dx):
    return torch.gradient(x, spacing=dx, edge_order=2)[0]


def plot_height_profile_evolution(r, h_profiles, params, n_lines=5):
    """Plot the evolution of the height profile over time."""
    plt.figure(figsize=(6, 4))
    for i, h_t in enumerate(h_profiles[::50]):
        plt.plot(r * 1e-3, h_t * 1e-3, c="dimgrey")
    plt.plot(r * 1e-3, h_profiles[0] * 1e-3, c="k", label="h0")
    plt.plot(r * 1e-3, h_profiles[-1] * 1e-3, c="r", label="Final")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.title("Evolution of Droplet Height Profile Over Time")
    plt.tight_layout()
    plt.show()


def inspect(drop_model, h):
    # compute values we want to plot (one forward pass broken out in components)
    dh_dr = grad(h, drop_model.params.dr)
    curvature = drop_model.calc_curvature(h)
    d_curvature_dr = grad(curvature, drop_model.params.dr)
    pressure = drop_model.calc_pressure(h)
    dp_dr = grad(pressure, drop_model.params.dr)
    u_grid = drop_model.calc_u_velocity(h)
    integral_u_r = torch.trapezoid(
        drop_model.r.unsqueeze(1) * u_grid, dx=drop_model.params.dz, dim=1
    )
    grad_u_r = grad(integral_u_r, drop_model.params.dr)
    flow_dh_dt = drop_model.calc_flow_dh_dt(h)

    plt.plot(h / torch.max(torch.abs(h)), label="h")
    plt.plot(dh_dr / torch.max(torch.abs(dh_dr)), label="dh/dr")
    plt.legend()
    plt.show()

    plt.plot(curvature / torch.max(torch.abs(curvature)), label="curvature")
    plt.plot(
        d_curvature_dr / torch.max(torch.abs(d_curvature_dr)), label="d curvature / dr"
    )
    plt.legend()
    plt.show()

    plt.plot(pressure / torch.max(torch.abs(pressure)), label="pressure")
    plt.plot(dp_dr / torch.max(torch.abs(dp_dr)), label="dp/dr")
    plt.title(
        f"p max: {torch.max(torch.abs(pressure))}, std: {torch.std(pressure)}, \n pgrad max: {torch.max(torch.abs(dp_dr))}"
    )
    plt.legend()
    plt.show()

    plt.plot(integral_u_r / torch.max(torch.abs(integral_u_r)), label="integral_u_r")
    plt.plot(grad_u_r / torch.max(torch.abs(grad_u_r)), label="grad_u_r")
    plt.legend()
    plt.show()

    plt.plot(flow_dh_dt, label="flow_dh_dt")
    plt.legend()
    plt.show()


def plot_velocity(drop_model, h, center_mask=6, corner_mask=3):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)

    # u_grid[u_grid == 0.0] = torch.nan
    # w_grid[w_grid == 0.0] = torch.nan
    u_grid = set_nans_in_center(u_grid, center_mask)
    u_grid = set_nans_in_corners(u_grid, corner_mask)
    w_grid = set_nans_in_center(w_grid, center_mask)
    w_grid = set_nans_in_corners(w_grid, corner_mask)

    plt.figure(figsize=(6, 3))
    plt.imshow(
        u_grid.T,
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
    setup_tick_formatter(plt.gca())
    plt.gca().autoscale_view()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label=r"$u(r, z)$")
    plt.title("Radial Velocity")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot vertical velocity (w) with height profile overlay
    plt.figure(figsize=(6, 3))
    plt.imshow(
        w_grid.T,
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
    setup_tick_formatter(plt.gca())
    plt.gca().autoscale_view()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label=r"$w(r, z)$")
    plt.title("Vertical Velocity")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()


def flow_viz(drop_model, h, center_mask=8, corner_mask=4, log_mag=False):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)
    flow_magnitude = torch.sqrt(u_grid**2 + w_grid**2)
    if log_mag:
        flow_magnitude = torch.log(flow_magnitude)
    flow_magnitude = set_nans_in_center(flow_magnitude, center_mask)
    flow_magnitude = set_nans_in_corners(flow_magnitude, corner_mask)
    flow_magnitude[flow_magnitude == 0.0] = torch.nan

    plt.figure(figsize=(6, 3))
    plt.imshow(
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

    setup_tick_formatter(plt.gca())
    plt.gca().autoscale_view()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    if log_mag:
        cmap_label = r"log $||v(r, z)||$"
    else:
        cmap_label = r"$||v(r, z)||$"
    plt.colorbar(label=cmap_label)
    plt.title(r"$v(r, z)$")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()

def flow_viz_w_evap(drop_model, h, center_mask=8, corner_mask=4, log_mag=False):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)

    w_e = drop_model.calc_evap_dh_dt(h)
    dh_dr = grad(h, drop_model.params.dr)
    m_dot = -w_e * drop_model.params.rho / torch.sqrt(1 + torch.square(dh_dr))

    flow_magnitude = torch.sqrt(u_grid**2 + w_grid**2)
    if log_mag:
        flow_magnitude = torch.log(flow_magnitude)
    flow_magnitude = set_nans_in_center(flow_magnitude, center_mask)
    flow_magnitude = set_nans_in_corners(flow_magnitude, corner_mask)
    flow_magnitude[flow_magnitude == 0.0] = torch.nan

    plt.figure(figsize=(6, 3))
    plt.imshow(
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

    setup_tick_formatter(plt.gca())
    plt.gca().autoscale_view()
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    if log_mag:
        cmap_label = r"log $||v(r, z)||$"
    else:
        cmap_label = r"$||v(r, z)||$"
    plt.colorbar(label=cmap_label)
    

    import math

    def get_normals(r_in, h_in, m_dot, length=1):
        dr = torch.zeros_like(h_in)
        dh = torch.zeros_like(h_in)
        for idx in range(0, len(r_in)-1, 10):
            x0, y0, xa, ya = r_in[idx], h_in[idx], r_in[idx+1], h_in[idx+1]
            dx, dy = xa-x0, ya-y0
            norm = math.hypot(dx, dy) * 1/length * m_dot[idx] / max(m_dot)
            dx /= norm
            dy /= norm

            dr[idx] = dx
            dh[idx] = dy
        dr[-1] = -dr[0]
        dh[-1] = -dh[0]
        return dr, dh
    dx, dy = get_normals(drop_model.r, h, m_dot)
    
    # Calculating the gradient
    L=10 # gradient length
    grad_temp = torch.ones(2, drop_model.r.shape[0])
    grad_temp[0, :] = -2*drop_model.r
    grad_temp /= torch.linalg.norm(grad_temp, axis=0)  # normalizing to unit vector
    nx = torch.vstack((drop_model.r - L/2 * grad_temp[0], drop_model.r + L/2 * grad_temp[0]))
    ny = torch.vstack((h - L/2 * grad_temp[1], h + L/2 * grad_temp[1]))
    
    quiver_step = 2
    r_down = drop_model.r[::quiver_step]
    h_down = h[::quiver_step]
    dx_down = dx[::quiver_step]
    dy_down = dy[::quiver_step]
    nx_down = nx[:, ::quiver_step]
    ny_down = ny[:, ::quiver_step]

    plt.quiver(
        r_down,
        h_down,
        r_down - dy_down,
        h_down + dx_down,
        #nx_down[1,:],
        #ny_down[1,:],
        color="black",
        scale=50,
        width=0.003,
        headwidth=3,  # scale=100_000
    )

    plt.xlim([0,drop_model.params.r_grid])
    plt.title(r"$v(r, z)$")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()


    


def setup_tick_formatter(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(
        (-0, 0)
    )  # Force scientific notation for values between 10^-1 and 10^1

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Ensure the ticks are redrawn
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-0, 0))


def set_nans_in_center(x, center_mask_size):
    if center_mask_size == 0:
        return x
    middle_i = x.shape[0] // 2
    start_i = max(middle_i - center_mask_size // 2, 0)
    end_i = min(middle_i + center_mask_size // 2 + 1, x.shape[0])
    x[start_i:end_i] = torch.nan
    return x


def set_nans_in_corners(x, corner_mask_size=5):
    """
    Set values to NaN inward from the start and end non-zero indices along the first axis.
    """
    if corner_mask_size == 0:
        return x
    # non_zero_indices = torch.nonzero(x.any(dim=1), as_tuple=True)[0]
    mask = torch.abs(x) > 1e-8
    non_zero_indices = torch.where(mask)[0]
    if len(non_zero_indices) == 0:
        return x
    start_idx = non_zero_indices[0]
    end_idx = non_zero_indices[-1]
    mask_start = min(start_idx + corner_mask_size, x.shape[0])
    mask_end = max(end_idx - corner_mask_size, 0)
    x[start_idx:mask_start, :] = float("nan")
    x[mask_end : end_idx + 1, :] = float("nan")
    return x
