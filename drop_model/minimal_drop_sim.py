import numpy as np

def calc_curvature(params, r, z, field_vars, h):
    dh_dr = as_grad(h, params.dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    return curvature_term

def calc_pressure(params, r, z, field_vars, h):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    # using Diddens implementation
    # note, square root should be approximated as unity in the limit h -> 0
    curvature_term = calc_curvature(params, r, z, field_vars, h)
    d_curvature_dr = as_grad(curvature_term, params.dr)
    # pressure = -params.sigma * (1 / r) * d_curvature_dr
    pressure = -params.sigma * safe_div(r) * d_curvature_dr
    return pressure

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


r_c=1e-3,  # Radius of the droplet in meters
hmax0=5e-4,  # Initial droplet height at the center in meters
Nr=200,  # Number of radial points
Nz=110,  # Number of z-axis points
Nt=10,  # Number of time steps
dr=1e-3 / Nr,  # Radial grid spacing
dz=5e-4 / Nz,  # Vertical grid spacing
dt=1e-3,  # Time step size eg 1e-5
rho=1,  # Density of the liquid (kg/m^3) eg 1
w_e=0.0,  # -1e-3,  # Constant evaporation rate (m/s) eg 1e-4
sigma=0.072,  # Surface tension (N/m) eg 0.072
eta=1e-5,  # Viscosity (Pa*s) eg 1e-3
d_sigma_dr=0.0

r = np.linspace(0, r_c, Nr)  # r grid (not avoiding r=0)
z = np.linspace(0, hmax0, Nz)

h0 = hmax0 * (1 - (r/ r_c) ** 2)

u_grid = compute_u_velocity(params, r, z, field_vars, h)

# Integrate r * u over z from 0 to h for each radial position
integral_u_r = np.trapz(r[:, None] * u_grid, dx=params.dz, axis=1)
# TODO scaling??
integral_u_r *= params.dz
grad_u_r = as_grad(integral_u_r, params.dr)
# grad_u_r = np.gradient(integral_u_r, params.dr)
grad_u_r = gaussian_filter(grad_u_r, sigma=10)

# Calculate dh/dt as radial term plus evaporation rate
# radial_term = (-1/r) * grad_u_r
radial_term = -1 * safe_div(r) * grad_u_r
radial_term = pad_radial_term(radial_term)
dh_dt = radial_term + params.w_e