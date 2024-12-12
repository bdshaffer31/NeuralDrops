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

    # Antoine's Equation
    A: float  
    B: float
    C: float

    D: float #Diffusvity of Vapor
    Mw: float #Molecular weight of Vapor

    Rs: float #Gas Constant
    T: float #Temperature of drop exterior
    RH: float #Relative Humidity


def setup_polynomial_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
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


def run_forward_euler_simulation(model, h0, t_lin, post_fn=None):
    dt = t_lin[1] - t_lin[0]
    h_profiles = [h0.clone()]  # Use clone() instead of copy() for PyTorch tensors
    h = h0.clone()

    for t in t_lin:
        print(
            t.item(), end="\r"
        )  # Print the current time step, use .item() to get the value
        dh_dt = model.calc_dh_dt(h)  # Compute dh/dt using the model
        h = h + dt * dh_dt  # Forward Euler step
        if post_fn is not None:
            h = post_fn(h)
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

def smoothing_fn(x):
        return gaussian_blur_1d(x, sigma=10)


def fourier_projection(h, num_frequencies=10):
    fft_coeffs = torch.fft.fft(h)
    fft_coeffs[num_frequencies + 1 : -num_frequencies] = 0
    return torch.fft.ifft(fft_coeffs).real


def drop_fourier_projection(h_0, num_frequencies=20):
    """
    Apply Fourier projection only to the non-zero interior portion of h_0.
    """
    # Identify the non-zero region
    non_zero_indices = torch.nonzero(h_0, as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return h_0  # If there are no non-zero elements, return the original tensor
    start_idx = non_zero_indices[0]
    end_idx = non_zero_indices[-1] + 1
    h_0_nonzero = h_0[start_idx:end_idx]
    fft_coeffs = torch.fft.fft(h_0_nonzero)
    fft_coeffs[num_frequencies + 1 : -num_frequencies] = 0
    h_0_filtered = torch.fft.ifft(fft_coeffs).real
    h_0_filtered_full = h_0.clone()
    h_0_filtered_full[start_idx:end_idx] = h_0_filtered

    return h_0_filtered_full


def drop_polynomial_fit(h_0, degree=3):
    """
    Fit a polynomial of a given degree to the non-zero interior portion of h_0.
    """
    # Identify the non-zero region
    # non_zero_indices = torch.nonzero(h_0, as_tuple=True)[0]
    mask = torch.abs(h_0) > 1e-8
    non_zero_indices = torch.where(mask)[0]
    if len(non_zero_indices) == 0:
        return h_0  # If there are no non-zero elements, return the original tensor
    start_idx = non_zero_indices[0]  # + 1
    end_idx = non_zero_indices[-1] + 1
    h_0_nonzero = h_0[start_idx:end_idx]
    x_nonzero = torch.linspace(0, 1, steps=h_0_nonzero.shape[0], dtype=torch.float64)

    # Fit a polynomial of the given degree to the non-zero portion
    # Construct the Vandermonde matrix
    powers = torch.arange(degree + 1, dtype=torch.float64)
    A = x_nonzero.unsqueeze(1) ** powers
    coeffs, *_ = torch.linalg.lstsq(A, h_0_nonzero.unsqueeze(1))
    h_0_fitted = (A @ coeffs).squeeze(1)

    

    h_0_fitted_full = h_0.clone()
    h_0_fitted_full[start_idx:end_idx] = h_0_fitted

    shift = int(h_0.shape[0] // 2 - (end_idx + start_idx) / 2)
    h_shifted = torch.roll(h_0_fitted_full, shifts=shift)

    return h_shifted


def drop_center_polynomial_fit(h_0, degree=3, mask_size=10, fit_size=20):
    """
    Fit a polynomial of a given degree to a specified region around the midpoint of h_0,
    excluding a central window (mask) and using a limited number of points (fit_size).
    """
    # Identify the non-zero region
    non_zero_indices = torch.nonzero(h_0, as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return h_0  # If there are no non-zero elements, return the original tensor

    start_idx = non_zero_indices[0]
    end_idx = non_zero_indices[-1] + 1

    # Determine the midpoint of the non-zero region
    mid_idx = (start_idx + end_idx) // 2

    # Define the region to exclude (mask region)
    mask_start = mid_idx - mask_size // 2
    mask_end = mid_idx + mask_size // 2

    # Ensure mask region stays within bounds
    mask_start = max(mask_start, start_idx)
    mask_end = min(mask_end, end_idx)

    # Define the fit region: fit_size points on either side of the mask
    fit_start = max(start_idx, mask_start - fit_size)
    fit_end = min(end_idx, mask_end + fit_size)

    # Extract the portion of h_0 for fitting
    fit_indices = torch.cat(
        (torch.arange(fit_start, mask_start), torch.arange(mask_end, fit_end))
    )
    if len(fit_indices) == 0:
        return h_0  # If there's no data to fit, return the original tensor

    h_0_fit = h_0[fit_indices]
    x_fit = torch.linspace(0, 1, steps=h_0_fit.shape[0], dtype=torch.float64)

    # Fit a polynomial of the given degree
    powers = torch.arange(degree + 1, dtype=torch.float64)
    A = x_fit.unsqueeze(1) ** powers
    coeffs, *_ = torch.linalg.lstsq(A, h_0_fit.unsqueeze(1))

    # Evaluate the polynomial fit over the mask region
    mask_length = mask_end - mask_start
    x_mask = torch.linspace(0, 1, steps=mask_length, dtype=torch.float64)
    A_mask = x_mask.unsqueeze(1) ** powers
    h_0_fitted_mask = (A_mask @ coeffs).squeeze(1)

    # Replace the mask region with the fitted values
    h_0_smoothed = h_0.clone()
    h_0_smoothed[mask_start:mask_end] = h_0_fitted_mask

    return h_0_smoothed
