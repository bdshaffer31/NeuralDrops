import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter


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

    D: float #Diffusvity of Vapor
    Mw: float #Molecular weight of Vapor

    Rs: float #Gas Constant
    T: float #Temperature of drop exterior
    RH: float #Relative Humidity


def setup_grids(params: SimulationParams):
    """Set up the grid arrays and initial field values."""
    # Radial and vertical grids
    r = np.linspace(params.dr, params.r_c, params.Nr)  # r grid (avoiding r=0)
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

        m_dot_grid=np.zeros((params.Nr)),  # mass loss
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
    x_padded = np.pad(x, (1, 0), mode="edge")
    grad_x = np.gradient(x_padded, dx, edge_order=2)
    return grad_x[1:]


def central_diff_4th_order(x, dx):
    kernel = np.array([-1, 8, 0, -8, 1]) / (12 * dx)
    derivative = np.convolve(x, kernel, mode="same")
    return derivative


def five_point_derivative(data, dx):
    """Compute the first derivative using the five-point stencil with boundary handling."""
    n = len(data)
    derivative = np.zeros_like(data)

    # Interior points using the five-point stencil
    for i in range(2, n - 2):
        derivative[i] = (
            -data[i + 2] + 8 * data[i + 1] - 8 * data[i - 1] + data[i - 2]
        ) / (12 * dx)

    # Boundary points using one-sided stencils
    # Forward difference for the left boundary
    derivative[0] = (
        -25 * data[0] + 48 * data[1] - 36 * data[2] + 16 * data[3] - 3 * data[4]
    ) / (12 * dx)
    derivative[1] = (
        -3 * data[0] - 10 * data[1] + 18 * data[2] - 6 * data[3] + data[4]
    ) / (12 * dx)

    # Backward difference for the right boundary
    derivative[-1] = (
        25 * data[-1] - 48 * data[-2] + 36 * data[-3] - 16 * data[-4] + 3 * data[-5]
    ) / (12 * dx)
    derivative[-2] = (
        3 * data[-5] - 16 * data[-4] + 36 * data[-3] - 48 * data[-2] + 25 * data[-1]
    ) / (12 * dx)

    return derivative


def gaussian_mollified_derivative(data, dx, sigma=1.0):
    """
    Compute the first derivative using a 5-point Gaussian-mollified kernel.
    """
    # Generate Gaussian-mollified kernel for 5 points
    offsets = np.array([-2, -1, 0, 1, 2])
    gaussian_weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    gaussian_weights /= gaussian_weights.sum()  # Normalize weights

    # Compute finite difference weights (central difference)
    finite_diff_weights = np.array([-1, -8, 0, 8, 1]) / 12  # Standard 5-point stencil

    # Combine Gaussian smoothing with finite difference weights
    kernel = gaussian_weights * finite_diff_weights

    # Apply convolution with periodic boundary conditions
    derivative = np.convolve(data, kernel, mode="same") / dx

    # Handle boundaries explicitly (optional, depending on BCs)
    derivative[0] = np.dot(data[:5], kernel) / dx
    derivative[1] = np.dot(data[:5], kernel) / dx
    derivative[-1] = np.dot(data[-5:], kernel[::-1]) / dx
    derivative[-2] = np.dot(data[-5:], kernel[::-1]) / dx

    return derivative


def calc_curvature(params, r, z, field_vars, h):
    dh_dr = as_grad(h, params.dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    return curvature_term


def calc_curvature_v2(params, r, z, field_vars, h):
    dh_dr = as_grad(h, params.dr)
    d2h_dr2 = as_grad(dh_dr, params.dr)
    curvature_term = np.abs(d2h_dr2) / np.sqrt(1 + dh_dr**2) ** 3
    return curvature_term


def calc_pressure(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr = as_grad(curvature_term, params.dr)
    pressure = -params.sigma * (1 / r) * d_curvature_dr
    return pressure


def calc_pressure_v2(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    curvature_term = calc_curvature_v2(params, r, z, field_vars, h)
    pressure = curvature_term * params.sigma * 2
    return pressure


def compute_u_velocity(params, r, z, field_vars, h):
    """Compute radial velocity u(r, z, t) using the given equation."""
    u_grid = np.zeros_like(field_vars.u_grid)
    
    h_star = params.hmax0/100
    n = 3
    m = 2
    #dis_press = -params.sigma*np.square(params.theta_e)*(n-1)*(m-1)/(n-m)/(2*h_star)*((h_star/params.hmax0)**n-(h_star/params.hmax0)**m)

    pressure = calc_pressure(params, r, z, field_vars, h) #+ dis_press

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


def h_mask_grid(grid_data, h, z):
    masked_grid = np.array(grid_data)
    for i in range(masked_grid.shape[0]):
        h_r = h[i]
        for j in range(masked_grid.shape[1]):
            z_val = z[j]
            if z_val > h_r:
                masked_grid[i, j] = 0
    return masked_grid

def mass_loss (params,field_vars,r,theta):
    p_sat = 10**(params.A-params.B/(params.C+params.T-273.15)) #Antoine's Equation
    p_sat = p_sat/760.0*101325.0 # conversion (mmHg) to (Pa)

    R_c = params.r_c #TODO
    b_c = 1.0 #TODO
    J_delta = 0 #Contribution from nonhomogenous surface concentration (multi-phase drops)

    cosh_alpha = (r**2*np.cos(theta) + R_c*np.sqrt(np.square(R_c)-np.square(r)*np.square(np.sin(theta))))/(np.square(R_c)-np.square(r))

    integral = np.zeros_like(field_vars.m_dot_grid)
    for i in range(len(r)):
        integral[i] , err = integrate.quad(lambda x:
                                           np.tanh(np.pi*x/(2.0*np.pi-2.0*theta))/
                                           (np.cosh(np.pi*x/(2.0*np.pi-2.0*theta))*np.sqrt(np.cosh(x)-cosh_alpha[i]))
                                           , np.arccosh(cosh_alpha[i]) , np.inf)

    N_prime_alpha = np.sqrt(2)*np.power(np.sqrt((cosh_alpha+np.cos(theta))),3)
    J_c = np.pi*N_prime_alpha*integral/(2*np.sqrt(2)*np.square(np.pi-theta))
    J_c[-1]=J_c[-2]*2 # Avoid NaN at r = R_c

    J_term = (b_c - params.RH)*J_c + J_delta

    m_dot = params.D*params.Mw*p_sat/(params.Rs*params.T*R_c)*J_term
    return m_dot


def evap_velocity (params,m_dot,h):
    dh_dr = as_grad(h,params.dr)
    w_e = -m_dot/params.rho* np.sqrt(1 + np.square(dh_dr))
    return w_e


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
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
    # TODO scaling??
    integral_u_r *= params.dz
    grad_u_r = as_grad(integral_u_r, params.dr)
    grad_u_r = gaussian_filter(grad_u_r, sigma=10)

    # Calculate dh/dt as radial term plus evaporation rate
    radial_term = (-1 / r) * grad_u_r
    # TODO bcs
    radial_term[0] = radial_term[3]
    radial_term[1] = radial_term[3]
    radial_term[2] = radial_term[3]
    # TODO negative sign
    dh_dt = radial_term + params.w_e
    # dh_dt = -1 * radial_term + params.w_e

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


# TODO use actual params
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
    dh_dr = as_grad(h, params.dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = as_grad(curvature_term, params.dr)
    pressure = calc_pressure(params, r, z, field_vars, h)
    dp_dr = as_grad(pressure, params.dr)

    u_grid = compute_u_velocity(params, r, z, field_vars, h)
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
    grad_u_r = as_grad(integral_u_r, params.dr)
    radial_term = (-1 / r) * grad_u_r
    radial_term[0] = radial_term[3]
    radial_term[1] = radial_term[3]
    radial_term[2] = radial_term[3]
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
        Nr=501,  # Number of radial points
        Nz=111,  # Number of z-axis points
        Nt=100,  # Number of time steps
        dr=1e-3 / (501-1),  # Radial grid spacing
        dz=5e-4 / (111-1),  # Vertical grid spacing
        dt=5e-5,  # Time step size eg 1e-5
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        w_e=-1e-3, # -1e-3,  # Constant evaporation rate (m/s) eg 1e-4
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
    h_profiles = eval(params, r, z, field_vars, h_0.copy())

    # plot the velocity profile and
    inspect(params, r, z, field_vars, h_profiles[-1])
    plot_velocity(params, r, z, field_vars, h_profiles[-1])


if __name__ == "__main__":
    run()