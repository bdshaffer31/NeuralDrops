import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.optimize import root
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


def setup_grids(params: SimulationParams):
    """Set up the grid arrays and initial field values."""
    # Radial and vertical grids
    # r = np.linspace(params.dr, params.r_c, params.Nr)  # r grid (avoiding r=0)
    r = np.linspace(-params.r_c, params.r_c, params.Nr)  # r grid (avoiding r=0)
    # r = np.linspace(0, params.r_c, params.Nr)  # r grid (not avoiding r=0)
    z = np.linspace(0, params.hmax0, params.Nz)  # z grid

    # Initialize field arrays
    field_vars = FieldVariables(
        u_grid=np.zeros((params.Nr, params.Nz)),  # r velocity
        w_grid=np.zeros((params.Nr, params.Nz)),  # z velocity
        p_grid=np.zeros((params.Nr)),  # pressure
        eta_grid=params.eta * np.ones((params.Nr, params.Nz)),  # constant viscosity
        sigma_grid=params.sigma * np.ones((params.Nr)),  # constant surface tension
        rho_grid=params.rho * np.ones((params.Nr, params.Nz)),  # density
        diff_grid=params.rho * np.ones((params.Nr, params.Nz)),  # diffusivity
    )

    return r, z, field_vars


def setup_parabolic_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
    # setup up a polynomial initial drop profile
    drop_fraction = 1.0  # percent of r taken up with drop (vs 0)
    h = np.zeros_like(r)
    occupied_length = int(drop_fraction * len(r))
    h[:occupied_length] = h0 * (
        1 - (r[:occupied_length] / (drop_fraction * r_c)) ** order
    )
    return h

def setup_cap_initial_h_profile(r, h0, r_c):
    # setup a spherical cap initial height profile
    R = (r_c**2 + h0**2) / (2 * h0)
    theta = np.arccos(1 - h0 * R)
    h = np.sqrt((2.0 * R * (r + R) - np.square(r + R))) - (R - h0)

    return h

def as_grad(x, dx):
    """Axis symmetric gradient (left side neumann boundary condition)"""
    # not used
    x_padded = np.pad(x, (1, 0), mode="edge")
    grad_x = np.gradient(x_padded, dx, edge_order=2)
    return grad_x[1:]

def grad(x, dx):
    return np.gradient(x, dx, edge_order=2)

def calc_curvature(params, r, z, field_vars, h):
    dh_dr = grad(h, params.dr)
    # epsilon = 1e-6
    epsilon = 0.0
    curvature_term = (r * dh_dr) / (np.sqrt(1 + dh_dr**2) + epsilon)
    return curvature_term

# def safe_inv(r, epsilon=1e-6):
#     div_r = np.zeros_like(r)
#     mask = np.abs(r) > epsilon
#     div_r[mask] = 1 / r[mask]
#     div_r[~mask] = 1 / epsilon
#     return div_r

def safe_inv(r):
    return 1/ r

# def safe_inv(r, epsilon=1e-6):
#     safe_r = np.where(np.abs(r) > epsilon, r, epsilon)
#     return 1 / safe_r


def calc_pressure(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr = grad(curvature_term, params.dr)
    # pressure = -params.sigma * (1 / r) * d_curvature_dr
    pressure = -params.sigma * safe_inv(r) * d_curvature_dr
    return pressure


def compute_u_velocity(params, r, z, field_vars, h):
    """Compute radial velocity u(r, z, t) using the given equation."""
    u_grid = np.zeros_like(field_vars.u_grid)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = grad(pressure, params.dr)
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
    div_r = safe_inv(r)
    integrand = div_r[:, None] * d_ur_grid_dr
    for j in range(1, len(z)):  # ignore BC
        w_grid[:, j] = -1 * np.trapz(integrand[:, : j + 1], dx=params.dz, axis=1)
    w_grid = interp_h_mask_grid(w_grid, h, z)
    w_grid[0, :] = w_grid[1, :]  # hack to deal with numerical issues at r ~ 0
    return w_grid


def h_mask_grid(grid_data, h, z):
    masked_grid = np.array(grid_data)
    for i in range(masked_grid.shape[0]):
        h_r = h[i]
        for j in range(masked_grid.shape[1]):
            z_val = z[j]
            if z_val > h_r:
                masked_grid[i, j] = 0
    return masked_grid


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

def calculate_dh_dt(t, params, r, z, field_vars, h):
    """Calculate dh/dt from u velocities for integration with solve_ivp"""
    h = h.reshape(len(r))

    # Compute the radial velocity field u(r, z, t)
    u_grid = compute_u_velocity(params, r, z, field_vars, h)

    # Integrate r * u over z from 0 to h for each radial position
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1) + 1e-8
    integral_u_r *= params.dz
    grad_u_r = np.gradient(integral_u_r, params.dr)
    grad_u_r = gaussian_filter(grad_u_r, sigma=10)

    # Calculate dh/dt as radial term plus evaporation rate
    radial_term = -1 * safe_inv(r) * grad_u_r
    dh_dt = radial_term + params.w_e
    # dh_dt = radial_term + r**2 * params.w_e / params.dr**2 # spoof an interesting evap

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

def run_backward_euler_simulation(params, r, z, field_vars, h0):
    h_profiles = [h0.copy()]
    h = h0.copy()

    for t in range(params.Nt):
        print(t, end="\r")
        
        def implicit_function(h_next):
            dh_dt = calculate_dh_dt((t + 1) * params.dt, params, r, z, field_vars, h_next)
            return h_next - h - params.dt * dh_dt
        
        result = root(implicit_function, h, method='hybr', options={'maxfev': 20})

        if result.success:
            h = result.x
        else:
            print("did not converge")
            h = result.x
            # raise RuntimeError(f"Implicit solver failed to converge at time step {t}")
        h = np.maximum(h, 0)
        h_profiles.append(h.copy())

    h_profiles = np.array(h_profiles)
    return h_profiles


def run_BDF_simulation(params, r, z, field_vars, h0):
    # Define the wrapper ODE function for solve_ivp
    def ode_func(t, h):
        return calculate_dh_dt(t, params, r, z, field_vars, h)

    # solve of t span
    t_span = (0, params.Nt * params.dt)
    t_eval = np.linspace(0, t_span[1], params.Nt)  # store solution at

    sol = solve_ivp(
        ode_func, t_span, h0, method="BDF", t_eval=t_eval  # BDF for stiff problems
    )

    h_profiles = sol.y.T
    return h_profiles


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
    plt.show()


def eval(params, r, z, field_vars, h0):
    h_profiles = run_forward_euler_simulation(params, r, z, field_vars, h0)
    plot_height_profile_evolution(r, h_profiles, params)
    return h_profiles


def inspect(params, r, z, field_vars, h):
    dh_dr = grad(h, params.dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = grad(curvature_term, params.dr)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = grad(pressure, params.dr)
    dp_dr[1] = dp_dr[0] + (dp_dr[2] - dp_dr[0])/2

    u_grid = compute_u_velocity(params, r, z, field_vars, h)
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
    grad_u_r = grad(integral_u_r, params.dr)
    # grad_u_r = np.gradient(integral_u_r, params.dr)
    # radial_term = (-1 / r) * grad_u_r
    radial_term = -1 * safe_inv(r) * grad_u_r
    dh_dt = radial_term + params.w_e

    # plt.plot(h / np.max(np.abs(h)), label="h")
    plt.plot(h, label="h")
    # plt.plot(dh_dr / np.max(np.abs(dh_dr)), label="dh/dr")
    plt.plot(dh_dr, label="dh/dr")
    plt.legend()
    plt.show()

    plt.plot(curvature_term / np.max(np.abs(curvature_term)), label="curvature")
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

    plt.plot(radial_term, label="radial_term")
    plt.legend()
    plt.show()

    plt.plot(dh_dt, label="dh_dt")
    plt.legend()
    plt.show()

    plt.plot(h, label="h")
    plt.plot(h + params.dt * dh_dt, label="h_t+1")
    plt.legend()
    plt.show()


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


def run():
    # Define the simulation parameters
    params = SimulationParams(
        r_c=1e-3,  # Radius of the droplet in meters
        hmax0=5e-4,  # Initial droplet height at the center in meters
        Nr=204,  # Number of radial points
        Nz=110,  # Number of z-axis points
        Nt=1000,  # Number of time steps
        dr=2 * 5e-3 / (204-1),  # Radial grid spacing
        dz=3e-4 / (110-1),  # Vertical grid spacing
        dt=1e-2,  # Time step size eg 1e-5
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        w_e=-1e-5,  # -1e-3,  # Constant evaporation rate (m/s) eg 1e-4
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-5,  # Viscosity (Pa*s) eg 1e-3
        d_sigma_dr=0.0,  # Surface tension gradient
    )

    # Initialize the grids and field variables
    r, z, field_vars = setup_grids(params)

    # run simulation and plot final profile
    h_0 = setup_parabolic_initial_h_profile(
        r, 0.8*params.hmax0, params.r_c, drop_fraction=1.0, order=4
    )
    # h_0 = setup_cap_initial_h_profile(r, 0.8*params.hmax0, params.r_c)
    h_profiles = eval(params, r, z, field_vars, h_0.copy())

    # plot the velocity profile and
    inspect(params, r, z, field_vars, h_profiles[-1].copy())
    plot_velocity(params, r, z, field_vars, h_profiles[-1].copy())
    inspect(params, r, z, field_vars, h_profiles[0].copy())
    plot_velocity(params, r, z, field_vars, h_profiles[0].copy())


if __name__ == "__main__":
    run()
