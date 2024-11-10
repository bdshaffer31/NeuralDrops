import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def as_grad(x, dx):
    x[0] = x[1]
    return np.gradient(x, dx)

def calc_pressure(h, sigma, dr, epsilon=1e-6):
    """Compute the radial pressure gradient using the nonlinear curvature formula."""
    dh_dr = as_grad(h, dr)
    curvature_term = (r * dh_dr) / np.sqrt(1 + dh_dr**2)
    d_curvature_dr = as_grad(curvature_term, dr)
    pressure_gradient = -sigma * (1 / (r + epsilon)) * d_curvature_dr
    # pressure_gradient = -sigma * d_curvature_dr
    
    return pressure_gradient

def radial_velocity(h, sigma, sigma_gradient, eta, dr, nz):
    """Compute radial velocity u(r, z, t) using the given equation."""
    pressure = calc_pressure(h, sigma, dr)
    dp_dr = as_grad(pressure, dr)


    d_sigma_dr = sigma_gradient
    z_vals = np.linspace(0, np.max(h), nz)
    dz = np.max(h) / (nz - 1)
    u = np.zeros((len(r), len(z_vals)))
    
    for i in range(len(r)):
        h_r = h[i]
        
        for j, z in enumerate(z_vals):
            if z > h_r:
                u[i, j] = 0
            else:
                integrand = (-(dp_dr[i]) * (h_r - z_vals) + d_sigma_dr) / eta
                u[i, j] = np.trapz(integrand[:j+1], dx=dz)
    
    u = gaussian_filter(u, sigma=5)
    
    return u, z_vals

def update_height(h, r, sigma, sigma_gradient, eta, rho, w_e, dr, nz, dt):
    """Update the height profile based on the radial velocity and evaporation rate."""
    # compute u(r,z,t)
    u, z_vals = radial_velocity(h, sigma, sigma_gradient, eta, dr, nz)
    
    # Integrate r * u over z from 0 to h for each radial position
    integral_u = np.trapz(r[:, None] * u, dx=np.max(h)/nz, axis=1)
    
    grad_u = as_grad(integral_u, dr)

    radial_term = (-1 / (r + 1e-6)) * grad_u
    # radial_term = grad_u
    dh_dt = radial_term + w_e
    
    # Update height profile
    h_new = h + dt * dh_dt
    
    # Ensure non-negative height
    h_new = np.maximum(h_new, 0)
    h_new[-1] = 0.0
    
    return h_new


def plotting_eval():
    # Initialize height profile storage
    h = h0 * (1 - (r / R)**2)
    fraction = 1.0
    h = np.zeros_like(r)
    occupied_length = int(fraction * len(r))
    h[:occupied_length] = h0 * (1 - (r[:occupied_length] / (fraction * R))**2)
    h_profiles = [h.copy()]
    integral_u_list = []
    radial_term_list = []
    nz=500


    for t in range(Nt):
    # Compute the radial velocity field u(r, z)
        u, z_vals = radial_velocity(h, sigma, sigma_gradient, eta, dr, nz)
        plt.imshow(u.T, aspect="auto", origin='lower', extent=[0, R, 0, np.max(h)])
        plt.plot(r, h, color='red', linewidth=2, label="Height profile $h(r)$")
        plt.xlabel('Radius (m)')
        plt.ylabel('Height (m)')
        plt.colorbar()
        plt.title('u')
        plt.show()

        # Integrate r * u over z from 0 to h for each radial position
        integral_u = np.trapz(r[:, None] * u, dx=np.max(h)/nz, axis=1)
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
    h = h0 * (1 - (r / R)**2)
    fraction = 1.0
    h = np.zeros_like(r)
    occupied_length = int(fraction * len(r))
    h[:occupied_length] = h0 * (1 - (r[:occupied_length] / (fraction * R))**2)
    # Initialize height profile storage
    h_profiles = [h.copy()]

    # Time-stepping loop
    for t in range(Nt):
        h = update_height(h, r, sigma, sigma_gradient, eta, rho, w_e, dr, nz=100, dt=dt)
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

if __name__ == "__main__":
    # Define Parameters
    R = 1e-3        # Radius of the droplet in meters
    Nx = 500        # Number of radial points
    Nt = 100        # Number of time steps
    dr = R / Nx     # Radial grid spacing
    dt = 1e-3       # Time step size
    rho = 1         # Density of the liquid (kg/m^3) (should normalize out for constant density)
    w_e = -1e-6      # Constant evaporation rate (m/s) eg 1e-7
    sigma = 0.072   # Surface tension (N/m) 0.072 for water
    eta = 1e-3      # Viscosity (Pa*s) 1e-3 for water (use constant for now)
    sigma_gradient = 1e-5  # Constant surface tension gradient (for Marangoni effect) eg 1e-5

    # Radial grid (excluding r=0 to avoid singularity)
    r = np.linspace(dr, R, Nx)

    # Initial height profile as a semi-spherical cap
    h0 = 1e-4  # Initial droplet height at the center in meters
    # h = h0 * (1 - (r / R)**2)

    # plotting_eval()
    eval()