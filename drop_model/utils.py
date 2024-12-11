from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class SimulationParams:
    r_c: float  # Radius of the droplet in meters
    hmax0: float  # Initial droplet height at the center in meters
    Nr: int  # Number of radial points
    Nz: int  # Number of z-axis points
    dr: float  # Radial grid spacing
    dz: float  # Vertical grid spacing
    rho: float  # Density of the liquid (kg/m^3)
    sigma: float  # Surface tension (N/m)
    eta: float  # Viscosity (Pa*s)


def setup_parabolic_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
    # setup up a polynomial initial drop profile
    drop_fraction = 1.0  # percent of r taken up with drop (vs 0)
    h = torch.zeros_like(r)
    occupied_length = int(drop_fraction * len(r))
    h[:occupied_length] = h0 * (
        1 - (r[:occupied_length] / (drop_fraction * r_c)) ** order
    )
    return h


def setup_cap_initial_h_profile(r, h0, r_c):
    # setup a spherical cap initial height profile
    R = (r_c**2 + h0**2) / (2 * h0)
    # theta = torch.arccos(torch.tensor([1 - h0 * R]))
    h = torch.sqrt((2.0 * R * (r + R) - torch.square(r + R))) - (R - h0)

    return h


def run_forward_euler_simulation(model, h0, t_lin):
    dt = t_lin[1] - t_lin[0]
    h_profiles = [h0.clone()]  # Use clone() instead of copy() for PyTorch tensors
    h = h0.clone()

    for t in t_lin:
        print(
            t.item(), end="\r"
        )  # Print the current time step, use .item() to get the value
        dh_dt = model.calc_dh_dt(h)  # Compute dh/dt using the model
        h = h + dt * dh_dt  # Forward Euler step
        h = torch.clamp(h, min=0)  # Ensure non-negative height using torch.clamp
        h_profiles.append(h.clone())  # Append a clone of the current height profile

    # Stack the list of tensors into a single tensor
    h_profiles = torch.stack(h_profiles)
    return h_profiles


def run_forward_euler_simulation_numpy(model, h0, t_lin):
    import numpy as np

    dt = t_lin[1] - t_lin[0]
    h_profiles = [h0.copy()]
    h = h0.copy()

    for t in t_lin:
        print(t, end="\r")
        dh_dt = model.calc_dh_dt(h)
        h = h + dt * dh_dt  # Forward Euler step
        h = np.maximum(h, 0)  # Ensure non-negative height
        h_profiles.append(h.copy())

    h_profiles = np.array(h_profiles)
    return h_profiles


def gaussian_blur_1d(input_tensor, sigma):
    radius = round(4 * sigma)
    kernel_size = 2 * radius + 1

    x = torch.linspace(
        -radius,
        radius,
        steps=kernel_size,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    kernel = kernel.view(1, 1, -1)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    padding = radius
    input_padded = F.pad(input_tensor, (padding, padding), mode="reflect")
    blurred = F.conv1d(input_padded, kernel)

    return blurred.squeeze(0).squeeze(0)
