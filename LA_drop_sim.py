import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def setup_parabolic_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
    drop_fraction = 1.0 # percent of r taken up with drop (vs 0)
    h = np.zeros_like(r)
    occupied_length = int(drop_fraction * len(r))
    h[:occupied_length] = h0 * (1 - (r[:occupied_length] / (drop_fraction * r_c))**order)
    return h

def setup_cap_initial_h_profile(r, h0, r_c, drop_fraction=1.0):
    R = (r_c**2 + h0**2) / (2*h0)
    theta = np.arccos(1-h0*R)
    
    # Calculate the contact angle in radians and convert to degrees
    contact_angle_rad = np.arcsin(r_c / R)
    contact_angle_deg = np.degrees(contact_angle_rad)
    
    # Compute the height profile using the spherical cap formula
    h_unnorm = np.sqrt(R**2 - r**2)
    h_unnorm = h_unnorm - np.min(h_unnorm)
    h_norm = h_unnorm / np.max(h_unnorm)
    h = h0 * h_norm
    
    return h, contact_angle_deg

def as_grad(x, dx):
    """Axis symmetric gradient (left side neumann boundary condition)"""
    x_padded = np.pad(x, (1, 0), mode='edge')
    grad_x = np.gradient(x_padded, dx, edge_order=2)
    return grad_x[1:]

def calc_pressure(h, sigma, dr):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    dh_dr = as_grad(h, dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = as_grad(curvature_term, dr)
    pressure = -sigma * (1 / r) * d_curvature_dr
    return pressure

def compute_u_velocity(h, u_grid, z, sigma, d_sigma_dr, eta, dr, dz):
    """Compute radial velocity u(r, z, t) using the given equation."""
    pressure = calc_pressure(h, sigma, dr)
    dp_dr = as_grad(pressure, dr)
    
    for i in range(len(r)):
        h_r = h[i]
        for j, z_val in enumerate(z):
            if z_val > h_r:
                u_grid[i, j] = 0
            else:
                integrand = (-(dp_dr[i]) * (h_r - z) + d_sigma_dr) / eta
                u_grid[i, j] = np.trapz(integrand[:j+1], dx=dz)
    
    # TODO this is a huge issue, maybe solved by interpolating
    # intead of jsut grabbing binary boundary cells
    # could also be solved by tracking the boundary and moving the discretization
    # u_grid = gaussian_filter(u_grid, sigma=5)
    
    return u_grid

def compute_w_velocity(h, w_grid, u_grid, r, z, dr, dz):
    ur_grid = u_grid * r[:, None]
    d_ur_grid_dr = np.gradient(ur_grid, dr, axis=0)
    integrand = (1 / r[:, None]) * d_ur_grid_dr
    for j in range(1, len(z)): # ignore BC
        w_grid[:, j] = -1 * np.trapz(integrand[:,:j+1], dx=dz, axis=1)
    w_grid = h_mask_grid(w_grid, h, z)
    w_grid[0,:] = w_grid[1,:]
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

def update_height(h, r, sigma, d_sigma_dr, eta, rho, w_e, dr, dz, dt):
    """Update the height profile based on the radial velocity and evaporation rate."""
    # compute u(r,z,t)
    u_grid = compute_u_velocity(h, sigma, d_sigma_dr, eta, dr, dz)
    
    # Integrate r * u over z from 0 to h for each radial position
    integral_u_r = np.trapz(r[:, None] * u_grid, dx=dz, axis=1)
    
    grad_u_r = as_grad(integral_u_r, dr)

    radial_term = (-1 / r) * grad_u_r
    dh_dt = radial_term + w_e
    
    # Update height profile
    h_new = h + dt * dh_dt
    
    # Ensure non-negative height
    h_new = np.maximum(h_new, 0)
    # pin the right side?
    h_new[-1] = 0.0
    
    return h_new


def plotting_eval():
    h_profiles = [h.copy()]
    integral_u_list = []
    radial_term_list = []


    for t in range(Nt):
    # Compute the radial velocity field u(r, z)
        u_grid = compute_u_velocity(h, sigma, d_sigma_dr, eta, dr, dz)
        plt.imshow(u_grid.T, aspect="auto", origin='lower', extent=[0, r_c, 0, np.max(h)])
        plt.plot(r, h, color='red', linewidth=2, label="Height profile $h(r)$")
        plt.xlabel('Radius (m)')
        plt.ylabel('Height (m)')
        plt.colorbar()
        plt.title('u')
        plt.show()

        # Integrate r * u over z from 0 to h for each radial position
        integral_u = np.trapz(r[:, None] * u_grid, dx=dz, axis=1)
        plt.plot(integral_u)
        plt.title('integral_u')
        plt.show()

        grad_u = as_grad(integral_u, dr)
        plt.plot(grad_u)
        plt.title('grad_u')
        plt.show()
        
        # Compute the radial term in the height evolution equation
        # radial_term = (-1 / r) * grad_u
        radial_term = grad_u
        dh_dt = -1 * radial_term + w_e
        # radial_term[0] = 0
        
        # Update height profile
        h = h + dt * dh_dt
        
        # Ensure non-negative height
        h = np.maximum(h, 0)
        
        h_profiles.append(h)
        integral_u_list.append(integral_u)
        radial_term_list.append(radial_term)

    # Plot Height Profile Evolution
    plt.figure(figsize=(10, 6))
    for i, h_t in enumerate(h_profiles[::100]):
        plt.plot(r * 1e3, h_t * 1e6, label=f"t = {i * dt * 100:.2f} s")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (µm)")
    plt.title("Evolution of Droplet Height Profile Over Time")
    plt.legend()
    plt.show()

    # Plot Integral of Radial Velocity Term Evolution
    plt.figure(figsize=(10, 6))
    for i, integral_u in enumerate(integral_u_list[::100]):
        plt.plot(r * 1e3, integral_u, label=f"t = {i * dt * 100:.2f} s")
    plt.xlabel("Radius (mm)")
    plt.ylabel(r"Integral of Radial Velocity $\int_0^h r u \, dz$")
    plt.title("Evolution of Integral of Radial Velocity Over Time")
    plt.legend()
    plt.show()

    # Plot Radial Term Evolution
    plt.figure(figsize=(10, 6))
    for i, radial_term in enumerate(radial_term_list[::100]):
        plt.plot(r * 1e3, radial_term, label=f"t = {i * dt * 100:.2f} s")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Radial Term in Height Evolution")
    plt.title("Evolution of Radial Term in Height Equation Over Time")
    plt.legend()
    plt.show()

def eval():
    # Initialize height profile storage
    h_profiles = [h.copy()]

    # Time-stepping loop
    # TODO use a more stable solver (eg built in)
    # update height should only compute dh_dt and not apply it
    for t in range(Nt):
        h = update_height(h, r, sigma, d_sigma_dr, eta, rho, w_e, dr, dz, dt=dt)
        h_profiles.append(h.copy())

    # Plot the evolution of the height profile over time
    plt.figure(figsize=(10, 6))
    for i, h_t in enumerate(h_profiles[::100]):
        plt.plot(r * 1e3, h_t * 1e6, c='dimgrey')
    plt.plot(r * 1e3, h_profiles[-1] * 1e6, c='r', label="final")
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (µm)")
    plt.legend()
    plt.title("Evolution of Droplet Height Profile Over Time")
    # plt.legend()
    plt.show()

def inspect():
    dh_dr = as_grad(h, dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = as_grad(curvature_term, dr)
    pressure = calc_pressure(h, sigma, dr)
    dp_dr = as_grad(pressure, dr)

    plt.plot(h/np.max(np.abs(h)), label='h')
    plt.plot(dh_dr/np.max(np.abs(dh_dr)), label='dh_dr')
    plt.legend()
    plt.show()

    plt.plot(curvature_term/np.max(np.abs(curvature_term)), label='curve')
    plt.plot(d_curvature_dr/np.max(np.abs(d_curvature_dr)), label='dcurve_dr')
    plt.legend()
    plt.show()

    plt.plot(pressure/np.max(np.abs(pressure)), label='pressure')
    plt.plot(dp_dr/np.max(np.abs(dp_dr)), label='dp_dr')
    plt.legend()
    plt.show()


def plot_velocity(u_grid, w_grid, r, z):

    u_grid = compute_u_velocity(h, u_grid, z, sigma, d_sigma_dr, eta, dr, dz)
    w_grid = compute_w_velocity(h, w_grid, u_grid, r, z, dr, dz)

    plt.figure(figsize=(10, 6))
    plt.imshow(u_grid.T, aspect="auto", origin='lower', extent=[0, r_c, 0, np.max(z)], cmap="viridis")
    plt.plot(r, h, color='red', linewidth=2, label="Height profile $h(r)$")  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Radial velocity $u(r, z)$")
    plt.title("Radial Velocity Field and Height Profile")
    plt.legend()
    plt.show()

    # Plot vertical velocity (w) with height profile overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(w_grid.T, aspect="auto", origin='lower', extent=[0, r_c, 0, np.max(z)], cmap="viridis")
    plt.plot(r, h, color='red', linewidth=2, label="Height profile $h(r)$")  # Overlay height profile
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.colorbar(label="Vertical velocity $w(r, z)$")
    plt.title("Vertical Velocity Field and Height Profile")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Define Parameters
    r_c = 1e-3         # Radius of the droplet in meters
    hmax0 = 5e-4       # Initial droplet height at the center in meters
    Nr = 200           # Number of radial points
    Nz = 110           # Number of z axis points
    Nt = 100           # Number of time steps
    dr = r_c / Nr      # Radial grid spacing
    dz = hmax0 / Nz    # Radial grid spacing
    dt = 1e-3          # Time step size
    rho = 1            # Density of the liquid (kg/m^3) (should normalize out for constant density)
    w_e = -1e-4        # Constant evaporation rate (m/s) eg 1e-6
    sigma = 0.072      # Surface tension (N/m) 0.072 for water
    eta = 1e-3         # Viscosity (Pa*s) 1e-3 for water (use constant for now)
    d_sigma_dr = 1e-5  # Constant surface tension gradient (for Marangoni effect) eg 1e-5

    # setup grid and height profile

    # Radial grid (excluding r=0 to avoid singularity)
    r = np.linspace(dr, r_c, Nr)         # r grid
    z = np.linspace(0, hmax0, Nz)        # z grid

    # setup fields
    u_grid = np.zeros((Nr, Nz))          # r velocity
    w_grid = np.zeros((Nr, Nz))          # z velocity
    # not used field variables from here down
    p_grid = np.zeros((Nr))              # pressure
    eta_grid = eta * np.ones((Nr, Nz))   # constant viscosity
    sigma_grid = sigma * np.ones((Nr))   # constant surface tension
    rho_grid = rho * np.ones((Nr, Nz))   # density
    diff_grid = rho * np.ones((Nr, Nz))  # diffusivity D_ab

    # initialize height profile
    # recomend using 4th order for testing (since it is further from steady state)
    # h, theta = setup_cap_initial_h_profile(r, hmax0, r_c, drop_fraction=1.0)
    h = setup_parabolic_initial_h_profile(r, hmax0, r_c, drop_fraction=1.0, order=4)
    # h = hmax0 - (hmax0/r_c) * r
    
    # plotting_eval()
    # eval()
    inspect()
    plot_velocity(u_grid, w_grid, r, z)