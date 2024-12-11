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


def plot_velocity(drop_model, h):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)

    u_grid[u_grid == 0.0] = torch.nan
    w_grid[w_grid == 0.0] = torch.nan

    plt.figure(figsize=(6, 3))
    plt.imshow(
        u_grid.T,
        aspect="auto",
        origin="lower",
        extent=[
            -drop_model.params.r_c,
            drop_model.params.r_c,
            0,
            torch.max(drop_model.z),
        ],
        cmap="magma",
    )
    plt.plot(
        drop_model.r, h, color="k", linewidth=2, label="$h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Radial velocity $u(r, z)$")
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
            -drop_model.params.r_c,
            drop_model.params.r_c,
            0,
            torch.max(drop_model.z),
        ],
        cmap="magma",
    )
    plt.plot(
        drop_model.r, h, color="k", linewidth=2, label="$h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Vertical velocity $w(r, z)$")
    plt.title("Vertical Velocity")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()


def flow_viz(drop_model, h):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)
    flow_magnitude = torch.sqrt(u_grid**2 + w_grid**2)
    flow_magnitude[flow_magnitude == 0.0] = torch.nan

    plt.figure(figsize=(6, 3))
    plt.imshow(
        flow_magnitude.T,
        aspect="auto",
        origin="lower",
        extent=[
            -drop_model.params.r_c,
            drop_model.params.r_c,
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
        (-3, 3)
    )  # Use scientific notation for values between 10^-3 and 10^3

    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    # Add a single scale label for each axis
    plt.gca().ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="velocity magnitude $||v(r, z)||$")
    plt.title("Flow Field")
    plt.grid(False)
    plt.tight_layout()
    plt.legend()
    plt.show()
