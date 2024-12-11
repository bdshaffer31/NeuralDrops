import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter, median_filter


@dataclass
class FieldVariables:
    u_grid: np.ndarray  # Radial velocity grid
    w_grid: np.ndarray  # Vertical velocity grid
    p_grid: np.ndarray  # Pressure grid
    eta_grid: np.ndarray  # Viscosity grid
    sigma_grid: np.ndarray  # Surface tension grid
    rho_grid: np.ndarray  # Density grid
    diff_grid: np.ndarray  # Diffusivity grid

    m_dot_grid: np.ndarray  # Diffusivity grid


@dataclass
class SimulationParams:
    r_c: float  # Radius of the droplet in meters
    hmax0: float  # Initial droplet height at the center in meters
    Nr: int  # Number of radial points
    Nz: int  # Number of z-axis points
    Nt: int  # Number of time steps
    dr: float  # Radial grid spacing
    dz: float  # Vertical grid spacing
    dt: float  # Time step size
    rho: float  # Density of the liquid (kg/m^3)
    w_e: float  # Constant evaporation rate (m/s)
    sigma: float  # Surface tension (N/m)
    eta: float  # Viscosity (Pa*s)
    d_sigma_dr: float  # Surface tension gradient

    # Antoine's Equation
    A: float
    B: float
    C: float

    D: float  # Diffusvity of Vapor
    Mw: float  # Molecular weight of Vapor

    Rs: float  # Gas Constant
    T: float  # Temperature of drop exterior
    RH: float  # Relative Humidity


params = SimulationParams(
    r_c=1e-3,  # Radius of the droplet in meters
    hmax0=3e-4,  # Initial droplet height at the center in meters
    Nr=499,  # Number of radial points
    Nz=110,  # Number of z-axis points
    Nt=10,  # Number of time steps
    dr=2.0 * 1e-3 / (499),  # Radial grid spacing
    dz=5e-4 / (110),  # Vertical grid spacing
    dt=1e-3,  # Time step size eg 1e-5
    rho=1,  # Density of the liquid (kg/m^3) eg 1
    w_e=-1e-3,  # -1e-3,  # Constant evaporation rate (m/s) eg 1e-4
    sigma=0.072,  # Surface tension (N/m) eg 0.072
    eta=1e-3,  # Viscosity (Pa*s) eg 1e-3
    d_sigma_dr=0.0,  # Surface tension gradient
    A=8.07131,  # Antoine Equation (-)
    B=1730.63,  # Antoine Equation (-)
    C=233.4,  # Antoine Equation (-)
    D=2.42e-5,  # Diffusivity of H2O in Air (m^2/s)
    Mw=0.018,  # Molecular weight H2O vapor (kg/mol)
    Rs=8.314,  # Gas Constant (J/(K*mol))
    T=293.15,  # Ambient Temperature (K)
    RH=0.20,  # Relative Humidity (-)
)


def setup_grids(params: SimulationParams):
    """Set up the grid arrays and initial field values."""
    # Radial and vertical grids
    r = np.linspace(-params.r_c, params.r_c, params.Nr + 1)  # r grid (avoiding r=0)
    z = np.linspace(0, params.hmax0, params.Nz + 1)  # z grid

    # Initialize field arrays
    field_vars = FieldVariables(
        u_grid=np.zeros((params.Nr + 1, params.Nz + 1)),  # r velocity
        w_grid=np.zeros((params.Nr + 1, params.Nz + 1)),  # z velocity
        p_grid=np.zeros((params.Nr + 1)),  # pressure
        eta_grid=params.eta
        * np.ones((params.Nr + 1, params.Nz + 1)),  # constant viscosity
        sigma_grid=params.sigma * np.ones((params.Nr + 1)),  # constant surface tension
        rho_grid=params.rho * np.ones((params.Nr + 1, params.Nz + 1)),  # density
        diff_grid=params.rho * np.ones((params.Nr + 1, params.Nz + 1)),  # diffusivity
        m_dot_grid=np.zeros((params.Nr + 1)),  # mass loss
    )

    return r, z, field_vars


def setup_cap_initial_h_profile(r, h0, r_c):
    # setup a spherical cap initial height profile
    R = (r_c**2 + h0**2) / (2 * h0)
    theta = np.arccos(1 - h0 * R)
    h = np.sqrt((2.0 * R * (r + R) - np.square(r + R))) - (R - h0)

    return h


def setup_parabolic_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
    # setup up a polynomial initial drop profile
    drop_fraction = 1.0  # percent of r taken up with drop (vs 0)
    h = np.zeros_like(r)
    occupied_length = int(drop_fraction * len(r))
    h[:occupied_length] = h0 * (
        1 - (r[:occupied_length] / (drop_fraction * r_c)) ** order
    )
    return h


# def as_grad(x, dx):
# """Axis symmetric gradient (left side neumann boundary condition)"""
# x_padded = np.pad(x, (1, 0), mode="edge")
# grad_x = np.gradient(x_padded, dx, edge_order=2)
# return grad_x[1:]


def as_grad(x, dx):
    """Axis symmetric gradient (left side neumann boundary condition)"""
    grad_x = np.gradient(x, dx, edge_order=2)
    return grad_x


def grad(x, dx):
    return np.gradient(x, dx, edge_order=2)


def calc_curvature(params, r, z, field_vars, h):
    dh_dr = as_grad(h, params.dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    return curvature_term


def calc_pressure_corrected(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr = as_grad(curvature_term, params.dr)
    Laplace_pressure = -params.sigma * (1 / r) * d_curvature_dr

    h_star = params.hmax0 / 100
    n = 3
    m = 2
    theta_e = 2 * np.arctan(params.hmax0 / params.r_c) / np.pi * 180
    dis_press = (
        -params.sigma
        * np.square(theta_e)
        * (n - 1)
        * (m - 1)
        / (n - m)
        / (2 * h_star)
        * ((h_star / params.hmax0) ** n - (h_star / params.hmax0) ** m)
    )

    pressure = Laplace_pressure + dis_press
    return pressure


def calc_pressure(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr = as_grad(curvature_term, params.dr)
    pressure = -params.sigma * (1 / r) * d_curvature_dr
    return pressure


def calc_curvature_v2(params, r, z, field_vars, h):
    dh_dr = as_grad(h, params.dr)
    d2h_dr2 = as_grad(dh_dr, params.dr)
    curvature_term = np.abs(d2h_dr2) / np.sqrt(1 + dh_dr**2) ** 3
    return curvature_term


def calc_pressure_v2(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    curvature_term = calc_curvature_v2(params, r, z, field_vars, h)
    pressure = curvature_term * params.sigma * 2
    return pressure


def interp_h_mask_grid(grid_data, h, z):
    dz = z[1] - z[0]
    masked_grid = np.array(grid_data)
    for i in range(masked_grid.shape[0]):
        h_r = h[i]
        lower_index = np.searchsorted(z, h_r) - 1  # last index beneath boundary
        if 0 <= lower_index < len(z) - 1:
            z_below = z[lower_index]
            value_above = masked_grid[i, lower_index + 1]
            occupation_percent = (h_r - z_below) / dz
            masked_grid[i, lower_index + 1] = occupation_percent * value_above
            masked_grid[i, lower_index + 2 :] = 0
        elif 0 >= lower_index:
            masked_grid[i, :] = 0
    return masked_grid


def compute_u_velocity(params, r, z, field_vars, h):
    """Compute radial velocity u(r, z, t) using the given equation."""
    u_grid = np.zeros_like(field_vars.u_grid)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = as_grad(pressure, params.dr)
    for i in range(len(r)):
        h_r = h[i]
        for j, z_val in enumerate(z):
            integrand = (-(dp_dr[i]) * (h_r - z) + params.d_sigma_dr) / params.eta
            u_grid[i, j] = np.trapz(integrand[: j + 1], dx=params.dz)
    u_grid = interp_h_mask_grid(u_grid, h, z)

    return u_grid


def compute_w_velocity(params, r, z, field_vars, h):
    w_grid = np.zeros_like(field_vars.w_grid)
    ur_grid = field_vars.u_grid * r[:, None]
    d_ur_grid_dr = np.gradient(ur_grid, params.dr, axis=0)
    integrand = (1 / r[:, None]) * d_ur_grid_dr
    for j in range(1, len(z)):  # ignore BC
        w_grid[:, j] = -1 * np.trapz(integrand[:, : j + 1], dx=params.dz, axis=1)
    w_grid = interp_h_mask_grid(w_grid, h, z)
    w_grid[0, :] = w_grid[1, :]  # hack to deal with numerical issues at r ~ 0
    return w_grid


def calculate_dh_dt(t, params, r, z, field_vars, h):
    """Calculate dh/dt from u velocities for integration with solve_ivp"""
    h = h.reshape(len(r))

    # Compute the radial velocity field u(r, z, t)
    u_grid = compute_u_velocity(params, r, z, field_vars, h)

    # Integrate r * u over z from 0 to h for each radial position
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
    # TODO scaling??
    integral_u_r *= params.dz
    grad_u_r = as_grad(integral_u_r, params.dr)
    grad_u_r = gaussian_filter(grad_u_r, sigma=10)

    middle_point = int(np.floor(len(grad_u_r) / 2))
    grad_u_r[: middle_point - 1] *= -1

    # print(middle_point)
    # print(r[middle_point], r[middle_point-1])
    # print(grad_u_r[middle_point-5:middle_point+5])
    avg_val = (grad_u_r[middle_point] + grad_u_r[middle_point - 1]) / 2
    # print(middle_point)
    grad_u_r[middle_point] = avg_val
    grad_u_r[middle_point - 1] = avg_val

    # Calculate dh/dt as radial term plus evaporation rate
    radial_term = (-1 / r) * grad_u_r
    dh_dt = radial_term + params.w_e

    return dh_dt


def run_forward_euler_simulation(params, r, z, field_vars, h0):
    h_profiles = [h0.copy()]
    h = h0.copy()

    for t in range(params.Nt):
        print(t, end="\r")
        dh_dt = calculate_dh_dt(t * params.dt, params, r, z, field_vars, h)
        h = h + params.dt * dh_dt  # Forward Euler step
        h = np.maximum(h, 0)  # Ensure non-negative height
        h_profiles.append(h.copy())

    h_profiles = np.array(h_profiles)
    return h_profiles


def eval(params, r, z, field_vars, h0):
    h_profiles = run_forward_euler_simulation(params, r, z, field_vars, h0)
    return h_profiles


def plot_velocity(params, r, z, field_vars, h):
    field_vars.u_grid = compute_u_velocity(params, r, z, field_vars, h)
    field_vars.w_grid = compute_w_velocity(params, r, z, field_vars, h)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        field_vars.u_grid.T,
        aspect="auto",
        origin="lower",
        extent=[0, params.r_c, 0, np.max(z)],
        cmap="viridis",
    )
    plt.plot(
        r, h, color="red", linewidth=2, label="Height profile $h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Radial velocity $u(r, z)$")
    plt.title("Radial Velocity Field and Height Profile")
    plt.legend()
    plt.show()

    # Plot vertical velocity (w) with height profile overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(
        field_vars.w_grid.T,
        aspect="auto",
        origin="lower",
        extent=[0, params.r_c, 0, np.max(z)],
        cmap="viridis",
    )
    plt.plot(
        r, h, color="red", linewidth=2, label="Height profile $h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Vertical velocity $w(r, z)$")
    plt.title("Vertical Velocity Field and Height Profile")
    plt.legend()
    plt.show()


def main():
    # Initialize the grids and field variables
    r, z, field_vars = setup_grids(params)

    h = setup_cap_initial_h_profile(r, params.hmax0, params.r_c)

    # Setup plot styling
    plt.rcParams.update({"font.size": 16})
    plt.rcParams["axes.linewidth"] = 3

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, h)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$d2hdr2(r)$")

    # print(params.dr)
    # print(r)

    dh_dr = as_grad(h, params.dr)
    d2h_dr2 = as_grad(dh_dr, params.dr)

    R_large = (params.r_c**2 + params.hmax0**2) / (2 * params.hmax0)
    dh_dr_2 = (-r) / np.sqrt((2.0 * R_large * (r + R_large) - np.square(r + R_large)))
    d2h_dr2_2 = -np.square(R_large) / np.sqrt(np.square(R_large) - np.square(r)) ** 3

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, dh_dr)
    plt.plot(r, dh_dr_2)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$d2hdr2(r)$")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, d2h_dr2)
    plt.plot(r, d2h_dr2_2)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$d2hdr2(r)$")

    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = as_grad(curvature_term, params.dr)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, curvature_term)
    plt.xlabel(r"$r$")
    plt.ylabel(r"curvature_term")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, d_curvature_dr)
    plt.xlabel(r"$r$")
    plt.ylabel(r"d_curvature_dr")

    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = as_grad(pressure, params.dr)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, pressure)
    plt.xlabel(r"$r$")
    plt.ylabel(r"pressure")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, dp_dr)
    plt.xlabel(r"$r$")
    plt.ylabel(r"dp_dr")

    u_grid = compute_u_velocity(params, r, z, field_vars, h)
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
    grad_u_r = as_grad(integral_u_r, params.dr)

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, integral_u_r)
    plt.xlabel(r"$r$")
    plt.ylabel(r"integral_u_r")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, grad_u_r)
    plt.xlabel(r"$r$")
    plt.ylabel(r"grad_u_r")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        field_vars.u_grid.T,
        aspect="auto",
        origin="lower",
        extent=[-params.r_c, params.r_c, 0, np.max(z)],
        cmap="viridis",
    )
    plt.plot(
        r, h, color="red", linewidth=2, label="Height profile $h(r)$"
    )  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Radial velocity $u(r, z)$")
    plt.title("Radial Velocity Field and Height Profile")
    plt.legend()
    plt.show()

    radial_term = (-1 / r) * grad_u_r
    dh_dt = radial_term + params.w_e

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, radial_term)
    plt.xlabel(r"$r$")
    plt.ylabel(r"radial_term")

    # Plot
    plt.figure(figsize=(12, 7))
    plt.plot(r, dh_dt)
    plt.xlabel(r"$r$")
    plt.ylabel(r"dh_dt")

    h_profiles = eval(params, r, z, field_vars, h.copy())

    plt.figure(figsize=(10, 6))
    for i, h_t in enumerate(h_profiles[::50]):
        plt.plot(r * 1e-3, h_t * 1e-3, c="dimgrey")
    plt.plot(r * 1e-3, h_profiles[0] * 1e-3, c="k", label="h0")
    plt.plot(r * 1e-3, h_profiles[-1] * 1e-3, c="r", label="Final")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.title("Evolution of Droplet Height Profile Over Time")
    plt.show()


def minimal_example():
    r, z, field_vars = setup_grids(params)
    # h = setup_cap_initial_h_profile(r, 0.8*params.hmax0, params.r_c)
    h = setup_parabolic_initial_h_profile(r, 0.8 * params.hmax0, params.r_c, order=4)
    h_profiles = eval(params, r, z, field_vars, h.copy())

    plt.figure(figsize=(10, 6))
    for i, h_t in enumerate(h_profiles[::50]):
        plt.plot(r * 1e-3, h_t * 1e-3, c="dimgrey")
    plt.plot(r * 1e-3, h_profiles[0] * 1e-3, c="k", label="h0")
    plt.plot(r * 1e-3, h_profiles[-1] * 1e-3, c="r", label="Final")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (mm)")
    plt.legend()
    plt.title("Evolution of Droplet Height Profile Over Time")
    plt.show()

    plot_velocity(params, r, z, field_vars, h_profiles[0])
    plot_velocity(params, r, z, field_vars, h_profiles[-1])


if __name__ == "__main__":
    minimal_example()
