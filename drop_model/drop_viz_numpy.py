# import numpy as np
import matplotlib.pyplot as plt
from drop_model.pure_drop_model_numpy import PureDropModel
from drop_model.utils import run_forward_euler_simulation
import numpy as np


def grad(x, dx):
    return np.gradient(x, dx, edge_order=2)


def plot_height_profile_evolution(r, h_profiles, params, n_lines=5):
    """Plot the evolution of the height profile over time."""
    plt.figure(figsize=(10, 6))
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
    integral_u_r = np.trapz(
        drop_model.r[:, None] * u_grid, dx=drop_model.params.dz, axis=1
    )
    grad_u_r = grad(integral_u_r, drop_model.params.dr)
    flow_dh_dt = drop_model.calc_flow_dh_dt(h)

    plt.plot(h / np.max(np.abs(h)), label="h")
    plt.plot(dh_dr / np.max(np.abs(dh_dr)), label="dh/dr")
    plt.legend()
    plt.show()

    plt.plot(curvature / np.max(np.abs(curvature)), label="curvature")
    plt.plot(d_curvature_dr / np.max(np.abs(d_curvature_dr)), label="d curvature / dr")
    plt.legend()
    plt.show()

    plt.plot(pressure / np.max(np.abs(pressure)), label="pressure")
    plt.plot(dp_dr / np.max(np.abs(dp_dr)), label="dp/dr")
    plt.title(
        f"p max: {np.max(np.abs(pressure))}, std: {np.std(pressure)}, \n pgrad max: {np.max(np.abs(dp_dr))}"
    )
    plt.legend()
    plt.show()

    plt.plot(integral_u_r / np.max(np.abs(integral_u_r)), label="integral_u_r")
    plt.plot(grad_u_r / np.max(np.abs(grad_u_r)), label="grad_u_r")
    plt.legend()
    plt.show()

    plt.plot(flow_dh_dt, label="flow_dh_dt")
    plt.legend()
    plt.show()


def plot_velocity(drop_model, h):
    u_grid = drop_model.calc_u_velocity(h)
    w_grid = drop_model.calc_w_velocity(h, u_grid)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        u_grid.T,
        aspect="auto",
        origin="lower",
        extent=[-drop_model.params.r_c, drop_model.params.r_c, 0, np.max(drop_model.z)],
        cmap="viridis",
    )
    plt.plot(
        drop_model.r, h, color="red", linewidth=2, label="Height profile $h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Radial velocity $u(r, z)$")
    plt.title("Radial Velocity Field and Height Profile")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot vertical velocity (w) with height profile overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(
        w_grid.T,
        aspect="auto",
        origin="lower",
        extent=[-drop_model.params.r_c, drop_model.params.r_c, 0, np.max(drop_model.z)],
        cmap="viridis",
    )
    plt.plot(
        drop_model.r, h, color="red", linewidth=2, label="Height profile $h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Vertical velocity $w(r, z)$")
    plt.title("Vertical Velocity Field and Height Profile")
    plt.legend()
    plt.tight_layout()
    plt.show()
