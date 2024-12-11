import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter

import torch

import time

@dataclass
class FieldVariables:
    u_grid: torch.tensor  # Radial velocity grid
    w_grid: torch.tensor  # Vertical velocity grid
    p_grid: torch.tensor  # Pressure grid
    eta_grid: torch.tensor  # Viscosity grid
    sigma_grid: torch.tensor  # Surface tension grid
    rho_grid: torch.tensor  # Density grid
    diff_grid: torch.tensor  # Diffusivity grid

    m_dot_grid: torch.tensor  # Diffusivity grid


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

    D: float #Diffusvity of Vapor
    Mw: float #Molecular weight of Vapor

    Rs: float #Gas Constant
    T: float #Temperature of drop exterior
    RH: float #Relative Humidity

def setup_grids(params: SimulationParams):
    """Set up the grid arrays and initial field values."""
    # Radial and vertical grids
    r = torch.linspace(-params.r_c, params.r_c, params.Nr + 1)  # r grid (avoiding r=0)
    z = torch.linspace(0, 1.2*params.hmax0, params.Nz + 1)  # z grid

    # Initialize field arrays
    field_vars = FieldVariables(
        u_grid=torch.zeros(params.Nr  + 1 , params.Nz  + 1),  # r velocity
        w_grid=torch.zeros(params.Nr  + 1, params.Nz  + 1),  # z velocity
        p_grid=torch.zeros(params.Nr  + 1),  # pressure
        eta_grid=params.eta * torch.ones(params.Nr  + 1, params.Nz  + 1),  # constant viscosity
        sigma_grid=params.sigma * torch.ones(params.Nr  + 1),  # constant surface tension
        rho_grid=params.rho * torch.ones(params.Nr  + 1, params.Nz  + 1),  # density
        diff_grid=params.rho * torch.ones(params.Nr  + 1, params.Nz  + 1),  # diffusivity

        m_dot_grid=torch.zeros(params.Nr  + 1),  # mass loss
    )

    return r, z, field_vars

def setup_cap_initial_h_profile(r, h0, r_c):

    # setup a spherical cap initial height profile
    R = (np.square(r_c) + np.square(h0)) / (2 * h0)
    theta = np.arccos(1 - h0 * R)
    h = torch.sqrt((2.0 * R * (r + R) - torch.square(r + R))) - (R - h0)
    return h

def setup_parabolic_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=4):
    # setup up a polynomial initial drop profile
    drop_fraction = 1.0  # percent of r taken up with drop (vs 0)
    h = torch.zeros_like(r)
    occupied_length = int(drop_fraction * len(r))
    h[:occupied_length] = h0 * (
        1 - (r[:occupied_length] / (drop_fraction * r_c)) ** order
    )
    return h

def grad(x, dx):
    return torch.gradient(x, spacing = [dx], edge_order=2)

def calc_curvature(params, r, z, field_vars, h):
    dh_dr, = grad(h, params.dr)
    curvature_term = (r * dh_dr) / torch.sqrt(1 + torch.square(dh_dr))
    return curvature_term

def calc_pressure(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr, = grad(curvature_term, params.dr)
    Laplace_pressure = -params.sigma * (1 / r) * d_curvature_dr

    h_star = params.hmax0/90
    n = 3
    m = 2
    theta_e = 2*np.arctan(params.hmax0/params.r_c)
    dis_press = -params.sigma*np.square(theta_e)*(n-1)*(m-1)/(n-m)/(2*h_star)*(np.power(h_star/params.hmax0, n)-np.power(h_star/params.hmax0, m))
    return Laplace_pressure + dis_press

def calc_curvature_v2(params, r, z, field_vars, h):
    dh_dr, = grad(h, params.dr)
    d2h_dr2, = grad(dh_dr, params.dr)
    curvature_term = torch.abs(d2h_dr2) / torch.pow(torch.sqrt(1 + torch.square(dh_dr)),3)
    return curvature_term

def calc_pressure_v2(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    curvature_term = calc_curvature_v2(params, r, z, field_vars, h)
    pressure = curvature_term * params.sigma * 2
    return pressure

def interp_h_mask_grid(grid_data, h, z):
    dz = z[1] - z[0]
    masked_grid = torch.tensor(grid_data)
    for i in range(masked_grid.shape[0]):
        h_r = h[i]
        lower_index = torch.searchsorted(z, h_r) - 1  # last index beneath boundary
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
    u_grid = torch.zeros_like(field_vars.u_grid)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr, = grad(pressure, params.dr)
    for i in range(len(r)):
        h_r = h[i]
        for j, z_val in enumerate(z):
            integrand = (-(dp_dr[i]) * (h_r - z) + params.d_sigma_dr) / params.eta
            u_grid[i, j] = torch.trapz(integrand[: j + 1], dx=params.dz)
    u_grid = interp_h_mask_grid(u_grid, h, z)
    return u_grid

def vectorize_compute_u_velocity(params, r, z, field_vars, h):
    """Compute radial velocity u(r, z, t) using the given equation."""
    u_grid = torch.zeros_like(field_vars.u_grid)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr, = grad(pressure, params.dr)

    def int_z(h, dp_dr):
        for j, z_val in enumerate(z):
            integrand = (-(dp_dr) * (h - z) + params.d_sigma_dr) / params.eta
            u_grid[1, j] = torch.trapz(integrand[: j + 1], dx=params.dz)
        return u_grid

    u_grid = torch.func.vmap(int_z)(h, dp_dr)
    u_grid = interp_h_mask_grid(u_grid, h, z)
    return u_grid

def compute_w_velocity(params, r, z, field_vars, h):
    w_grid = torch.zeros_like(field_vars.w_grid)
    ur_grid = field_vars.u_grid * r[:, None]
    d_ur_grid_dr = torch.gradient(ur_grid, params.dr, axis=0)
    integrand = (1 / r[:, None]) * d_ur_grid_dr
    for j in range(1, len(z)):  # ignore BC
        w_grid[:, j] = -1 * torch.trapz(integrand[:, : j + 1], dx=params.dz, axis=1)
    w_grid = interp_h_mask_grid(w_grid, h, z)
    w_grid[0, :] = w_grid[1, :]  # hack to deal with numerical issues at r ~ 0
    return w_grid

def mass_loss_tracked_r_c(params, r , h):

    def mass_loss (params, r_in, theta, R_c):
        p_sat = 10**(params.A-params.B/(params.C+params.T-273.15)) #Antoine's Equation
        p_sat = p_sat/760.0*101325.0 # conversion (mmHg) to (Pa)
        #print(p_sat)

        #R_c = params.r_c #TODO
        b_c = 1.0 #TODO
        J_delta = 0 #Contribution from nonhomogenous surface concentration (multi-phase drops)

        lam = -(np.pi-2.0*theta)/(2.0*np.pi-2.0*theta)
        J_c = torch.pow(1-torch.square(r_in/R_c),lam)

        J_c[-1]=J_c[-2]
        J_c[0]=J_c[1]

        J_term = (b_c - params.RH)*J_c + J_delta

        m_dot = params.D*params.Mw*p_sat/(params.Rs*params.T*R_c)*J_term
        return m_dot
    
    h_max_current = torch.max(h)
    num_c = list(map(lambda i: i > 0.01 * params.hmax0, h)).index(True)
    r_c_current = - r[num_c]
    theta_current = 2*torch.arctan(h_max_current/r_c_current)
    
    m_dot = torch.zeros_like(r)
    m_dot[num_c:-num_c] = mass_loss(params, r[num_c:-num_c], theta_current, r_c_current)
    
    return m_dot

def evap_velocity (params, m_dot, h):
    dh_dr, = grad(h,params.dr)
    w_e = -m_dot/params.rho* torch.sqrt(1 + torch.square(dh_dr))
    return w_e

def calculate_dh_dt(t, params, r, z, field_vars, h):
    """Calculate dh/dt from u velocities for integration with solve_ivp"""
    h = h.reshape(len(r))

    # Compute the radial velocity field u(r, z, t)
    u_grid = compute_u_velocity(params, r, z, field_vars, h)

    # Integrate r * u over z from 0 to h for each radial position
    integral_u_r = torch.trapz(r[:, None] * u_grid, dx=params.dz, axis=1) + 1.0e-8
    # TODO scaling??
    integral_u_r *= params.dz
    grad_u_r, = grad(integral_u_r, params.dr)
    grad_u_r = gaussian_filter(grad_u_r, sigma=10)

    # Calculate dh/dt as radial term plus evaporation rate
    radial_term = (-1 / r) * grad_u_r

    m_dot = mass_loss_tracked_r_c(params, r , h)
    w_e = evap_velocity(params, m_dot, h)

    dh_dt = radial_term + w_e

    return dh_dt

def run_forward_euler_simulation(params, r, z, field_vars, h0):
    h_profiles = [h0]
    h = h0

    for t in range(params.Nt):
        print(t, end="\r")
        dh_dt = calculate_dh_dt(t * params.dt, params, r, z, field_vars, h)
        h = h + params.dt * dh_dt  # Forward Euler step
        #h = torch.maximum(h, 0)  # Ensure non-negative height
        h_profiles.append(h)

    h_profiles = np.array(h_profiles)
    return h_profiles

def plot_height_profile_evolution(r, h_profiles, params, n_lines=5):
    """Plot the evolution of the height profile over time."""
    plt.figure(figsize=(10, 6))
    for i, h_t in enumerate(h_profiles[::20]):
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
    dh_dr, = grad(h, params.dr)
    curvature_term = (r * dh_dr) / torch.sqrt(1 + dh_dr**2)
    d_curvature_dr, = grad(curvature_term, params.dr)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = grad(pressure, params.dr)

    u_grid = compute_u_velocity(params, r, z, field_vars, h)
    integral_u_r = torch.trapz(r[:, None] * u_grid, dx=params.dz, axis=1) +1.0e-8
    grad_u_r, = grad(integral_u_r, params.dr)
    radial_term = (-1 / r) * grad_u_r

    m_dot = mass_loss_tracked_r_c(params, r , h)
    w_e = evap_velocity(params, m_dot, h)

    dh_dt = radial_term + w_e

    plt.plot(h, label="h")
    plt.legend()
    plt.show()

    plt.plot(dh_dr, label="dh/dr")
    plt.legend()
    plt.show()

    plt.plot(curvature_term, label="curvature")
    plt.legend()
    plt.show()

    plt.plot(d_curvature_dr, label="d curvature / dr")
    plt.legend()
    plt.show()

    plt.plot(pressure, label="pressure")
    plt.title(
        f"p max: {torch.max(torch.abs(pressure))}, std: {torch.std(pressure)}"
    )
    plt.legend()
    plt.show()

    plt.plot(dp_dr, label="dp/dr")
    plt.title(
        f"p max: {torch.max(torch.abs(pressure))}, std: {torch.std(pressure)}, \n pgrad max: {torch.max(torch.abs(dp_dr))}"
    )
    plt.legend()
    plt.show()

    plt.plot(integral_u_r, label="integral_u_r")
    plt.legend()
    plt.show()

    plt.plot(grad_u_r, label="grad_u_r")
    plt.legend()
    plt.show()

    plt.plot(radial_term, label="radial_term")
    plt.legend()
    plt.show()

    plt.plot(m_dot, label="mass_loss")
    plt.legend()
    plt.show()

    plt.plot(w_e, label="Evap_velocity")
    plt.legend()
    plt.show()

    plt.plot(dh_dt, label="dh_dt")
    plt.legend()
    plt.show()

    plt.plot(h, label="h")
    plt.plot(h + params.dt * dh_dt, label="h_t+1")
    plt.legend()
    plt.show()


def time_funcs(params, r, z, field_vars, h):
    Start = time.perf_counter()
    dh_dr ,  = grad(h, params.dr)
    End = time.perf_counter()
    print(f"grad runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    curvature_term = (r * dh_dr) / torch.sqrt(1 + torch.square(dh_dr))
    End = time.perf_counter()
    print(f"curvature_term runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    d_curvature_dr, = grad(curvature_term, params.dr)
    End = time.perf_counter()
    print(f"d_curvature_dr runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    pressure = calc_pressure(params, r, z, field_vars, h)
    End = time.perf_counter()
    print(f"calc_pressure runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    dp_dr, = grad(pressure, params.dr)
    End = time.perf_counter()
    print(f"dp_dr runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    u_grid = compute_u_velocity(params, r, z, field_vars, h)
    End = time.perf_counter()
    print(f"u_grid runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    integral_u_r = torch.trapz(r[:, None] * u_grid, dx=params.dz, axis=1) +1.0e-8
    End = time.perf_counter()
    print(f"integral_u_r runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    grad_u_r, = grad(integral_u_r, params.dr)
    End = time.perf_counter()
    print(f"integral_u_r runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    radial_term = (-1 / r) * grad_u_r
    End = time.perf_counter()
    print(f"radial_term runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    m_dot = mass_loss_tracked_r_c(params, r , h)
    End = time.perf_counter()
    print(f"m_dot runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    w_e = evap_velocity(params, m_dot, h)
    End = time.perf_counter()
    print(f"w_e runtime: {(End - Start)*1000} ms\n")

    Start = time.perf_counter()
    dh_dt = radial_term + w_e
    End = time.perf_counter()
    print(f"dh_dt runtime: {(End - Start)*1000} ms\n")


def plot_velocity(params, r, z, field_vars, h):
    field_vars.u_grid = compute_u_velocity(params, r, z, field_vars, h)
    field_vars.w_grid = compute_w_velocity(params, r, z, field_vars, h)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        field_vars.u_grid.T,
        aspect="auto",
        origin="lower",
        extent=[-params.r_c, params.r_c, 0, torch.max(z)],
        cmap="viridis", vmin = 0.0, vmax = 0.02,
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
        extent=[-params.r_c, params.r_c, 0, torch.max(z)],
        cmap="viridis",  vmin = 0.0, vmax = 0.01,
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
        hmax0=3e-4,  # Initial droplet height at the center in meters
        Nr=203,  # Number of radial points
        Nz=111,  # Number of z-axis points
        Nt=20,  # Number of time steps
        dr= 2.0 * 1e-3 / 203,  # Radial grid spacing
        dz=3e-4 / 111,  # Vertical grid spacing
        dt=1e-4,  # Time step size eg 1e-5
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        w_e=-1e-4, # -1e-3,  # Constant evaporation rate (m/s) eg 1e-4
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-3,  # Viscosity (Pa*s) eg 1e-3
        d_sigma_dr=0.0,  # Surface tension gradient

        A = 8.07131, # Antoine Equation (-)
        B = 1730.63, # Antoine Equation (-)
        C = 233.4, # Antoine Equation (-)
        D = 2.42e-5, # Diffusivity of H2O in Air (m^2/s)
        Mw = 0.018, # Molecular weight H2O vapor (kg/mol)
        Rs = 8.314, # Gas Constant (J/(K*mol))
        T = 293.15, # Ambient Temperature (K)
        RH = 0.20, # Relative Humidity (-)
    )

    # Initialize the grids and field variables
    r, z, field_vars = setup_grids(params)

    # run simulation and plot final profile
    h_0 = setup_cap_initial_h_profile(
        r, params.hmax0, params.r_c
    )
    time_funcs(params, r, z, field_vars, h_0)

    h_profiles = eval(params, r, z, field_vars, h_0)

    # plot the velocity profile and
    #inspect(params, r, z, field_vars, h_profiles[-1])
    plot_velocity(params, r, z, field_vars, h_profiles[-1])


if __name__ == "__main__":
    run()