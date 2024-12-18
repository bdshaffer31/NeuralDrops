from dataclasses import dataclass
import torch
import torch.nn.functional as F
from functorch import vmap


@dataclass
class SimulationParams:
    r_grid: float  # Radius of the droplet in meters
    hmax0: float  # Initial droplet height at the center in meters
    Nr: int  # Number of radial points
    Nz: int  # Number of z-axis points
    dr: float  # Radial grid spacing
    dz: float  # Vertical grid spacing
    rho: float  # Density of the liquid (kg/m^3)
    sigma: float  # Surface tension (N/m)
    eta: float  # Viscosity (Pa*s)


@dataclass
class EvapParams:
    # Antoine's Equation
    A: float
    B: float
    C: float
    D: float  # Diffusvity of Vapor
    Mw: float  # Molecular weight of Vapor
    Rs: float  # Gas Constant
    T: float  # Temperature of drop exterior
    RH: float  # Relative Humidity


def setup_polynomial_initial_h_profile(r, h0, r_c, drop_fraction=1.0, order=2):
    # setup up a polynomial initial drop profile
    drop_fraction = 1.0  # percent of r taken up with drop (vs 0)
    h = torch.zeros_like(r)
    num_c = list(map(lambda i: i > -r_c, r)).index(True) + 1
    h[num_c:-(num_c)] = h0 * (1 - (r[num_c:-(num_c)] / (drop_fraction * r_c)) ** order)
    return h


def setup_cap_initial_h_profile(r_cap, h0, r_c):
    h = torch.zeros_like(r_cap)
    num_c = list(map(lambda i: i >= -r_c, r_cap)).index(True) + 1
    # setup a spherical cap initial height profile
    R = (r_c**2 + h0**2) / (2 * h0)
    # theta = torch.arccos(torch.tensor([1 - h0 * R]))
    h[num_c:-(num_c)] = torch.sqrt(
        (
            2.0 * R * (r_cap[num_c:-(num_c)] + R)
            - torch.square(r_cap[num_c:-(num_c)] + R)
        )
    ) - (R - h0)
    # h = torch.sqrt((2.0 * R * (r + R) - torch.square(r + R))) - (R - h0)

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
        h = torch.maximum(h, torch.tensor(0.0))  # Ensure non-negative height
        if post_fn is not None:
            h = post_fn(h)
        h_profiles.append(h.clone())  # Append a clone of the current height profile

    # Stack the list of tensors into a single tensor
    h_profiles = torch.stack(h_profiles)
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
    #A = A.to(torch.float32)
    A = A.to(torch.float64)
    #print(h_0_nonzero.shape)
    h_coeffs = h_0_nonzero.unsqueeze(1)
    #print(h_coeffs.shape)
    coeffs, *_ = torch.linalg.lstsq(A, h_coeffs)
    h_0_fitted = (A @ coeffs).squeeze(1)

    h_0_fitted_full = h_0.clone()
    h_0_fitted_full[start_idx:end_idx] = h_0_fitted

    shift = int(h_0.shape[0] // 2 - (end_idx + start_idx) / 2)
    h_shifted = torch.roll(h_0_fitted_full, shifts=shift)

    return h_shifted


def drop_polynomial_fit_batch_vmap(h_batch, degree=3):
    # DOES NOT WORK - many issues
    return vmap(lambda h_0: drop_polynomial_fit(h_0, degree))(h_batch)


def drop_polynomial_fit_batch(h_batch, degree=3):
    # oh well ...
    h_fit = torch.zeros_like(h_batch)
    for i, h in enumerate(h_batch):
        h_fit[i] = drop_polynomial_fit(h, degree)
    return h_fit


def symmetrize(x):
    is_non_batched = x.dim() == 1
    if is_non_batched:
        x = x.unsqueeze(0)
    half_len = x.size(1) // 2
    x[:, :half_len] = torch.flip(x[:, half_len:], dims=[1])
    if is_non_batched:
        x = x.squeeze(0)
    return x


def load_drop_params_from_data(data):
    first_key = list(data.keys())[0]
    t_lin = data[first_key]["t_lin"]
    r_lin = data[first_key]["r_lin"]
    z_lin = data[first_key]["z_lin"]
    dt = t_lin[1] - t_lin[0]
    dr = r_lin[1] - r_lin[0]
    dz = z_lin[1] - z_lin[0]

    params = SimulationParams(
        r_grid=torch.max(r_lin),
        hmax0=torch.max(z_lin),
        Nr=r_lin.shape[0],
        Nz=z_lin.shape[0],
        dr=dr,
        dz=dz,
        # defining these here just creates opportunities to mess them up probably
        rho=1,
        sigma=0.072,
        eta=1e-3,
    )
    return params, dt


def dis_press(params, r_grid, hmax0, sigma):
    h_star = hmax0 / 100
    n = 3
    m = 2
    theta_e = 2 * torch.arctan(torch.tensor(params.hmax0 / (0.5 * r_grid)))
    dis_press = (
        -sigma
        * torch.square(torch.tensor(theta_e))
        * (n - 1)
        * (m - 1)
        / (n - m)
        / (2 * h_star)
        * (
            torch.pow(torch.tensor(h_star / hmax0), n)
            - torch.pow(torch.tensor(h_star / hmax0), m)
        )
    )
    return dis_press
