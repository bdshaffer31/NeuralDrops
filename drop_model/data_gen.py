import matplotlib.pyplot as plt
import torch
import numpy as np

from pure_drop_model import PureDropModel
import utils as utils
import drop_viz as drop_viz
import evap_models as evap_models


def default_params():
    params = utils.SimulationParams(
        r_grid=1.28e-3,  # Radius of the droplet in meters
        hmax0=7.0e-4,  # Initial droplet height at the center in meters
        Nr=640,  # Number of radial points
        Nz=220,  # Number of z-axis points
        dr=2 * 1.28e-3 / (640 - 1),  # Radial grid spacing
        dz=7.0e-4 / (220 - 1),  # Vertical grid spacing
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-3,  # Viscosity (Pa*s) eg 1e-5
    )
    return params

def deegan_params():
    evap_params = utils.EvapParams(
        A=8.07131,  # Antoine Equation (-)
        B=1730.63,  # Antoine Equation (-)
        C=233.4,  # Antoine Equation (-)
        D=2.42e-5,  # Diffusivity of H2O in Air (m^2/s)
        Mw=0.018,  # Molecular weight H2O vapor (kg/mol)
        # Rs = 8.314, # Gas Constant (J/(K*mol))
        Rs=461.5,  # Gas Constant (J/(K*kg))
        T=293.15,  # Ambient Temperature (K)
        RH=0.20,  # Relative Humidity (-)
    )
    return evap_params


def run_sim(h0, evap_params, params, t_lin):
    def constant_evap_model(params, r, h, kappa=1.0e-5):
        sqr = torch.linspace(-1, 1, len(r)) ** 2
        return -kappa * torch.ones_like(h) * sqr
    
    #evap_params = deegan_params()
    def deegan_evap_wrapped(evap_params, params, r, h):
        return evap_models.deegan_evap_model(evap_params, params, r, h)

    def smoothing_fn(x):
        return utils.gaussian_blur_1d(x, sigma=10)

    def post_fn(h):
        h = torch.clamp(h, min=0)  # ensure non-negative height
        h = utils.drop_polynomial_fit(h, 8)  # project height on polynomial basis
        return h

    drop_model = PureDropModel(
        params, evap_params=evap_params, evap_model=deegan_evap_wrapped, smoothing_fn=smoothing_fn
    )

    h_history = utils.run_forward_euler_simulation(drop_model, h0, t_lin, post_fn)

    return h_history


def polynomial_init(r, r_c, hmax0, alpha, beta, gamma, epsilon):
    y = torch.zeros_like(r)
    num_c = list(map(lambda i: i > -r_c, r)).index(True) + 1
    y[num_c:-(num_c)] = 1 - (alpha * (r[num_c:-(num_c)]/r_c)**2 + beta * (r[num_c:-(num_c)]/r_c)**4 + gamma * (r[num_c:-(num_c)]/r_c)**6 + epsilon * (r[num_c:-(num_c)]/r_c)**8)
    y -= torch.min(y[num_c:-(num_c)])
    y /= torch.max(y[num_c:-(num_c)])
    y *= hmax0
    y[0:num_c] = 0.0
    y[-num_c:] = 0.0
    return y

def cap_init(r_cap, h0, r_c):
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

    return h

def main():
    drop_viz.set_styling()
    torch.set_default_dtype(torch.float64)

    params = default_params()
    evap_params = deegan_params()
    r_lin = torch.linspace(-params.r_grid, params.r_grid, params.Nr)
    z_lin = torch.linspace(0, params.hmax0, params.Nz)
    x_lin = torch.linspace(-1, 1, params.Nr)
    Nt = 26000
    dt = 2e-3
    t_lin = torch.linspace(0, dt * Nt, Nt)
    r_c = 0.8
    h_init = 0.5*params.hmax0

    for i in range(1):
        alpha = np.random.rand()
        beta = np.random.rand()
        gamma = np.random.rand()
        y = polynomial_init(x_lin, r_c, h_init, alpha, beta, gamma, 0.0)
        #y = cap_init(x_lin, h_init, r_c)
        plt.plot(r_lin, y, alpha=0.2, c="k")
    plt.show()

    # generate the datasets
    results = {}
    for i in range(1):
        if i > 10:
            params.T = 303.15

        if i > 1:
            params.hmax0 = 1.1 * params.hmax0


        alpha = np.random.rand()
        beta = np.random.rand()
        gamma = np.random.rand()
        h0 = polynomial_init(x_lin, r_c, h_init, alpha, beta, gamma, 0.0)
        #h0 = cap_init(x_lin, h_init, r_c)
        h_history = run_sim(h0, evap_params, params, t_lin)[:-1]

        #plt.plot(h_history[0])
        #plt.plot(h_history[-1])
        #plt.show()
        print(torch.min(h_history), torch.max(h_history))
        drop_viz.plot_height_profile_evolution(r_lin, h_history[::100], params)

        print(h_history.shape, t_lin.shape)
        print(h_history[::25].shape, t_lin[::25].shape)

        # Saved dataset MUST have:
        # "profile": profile,
        # "t": t_lin,
        # "r": r_lin,
        # "z": z_lin,
        # and can additionally have any number of extra contents or conditioning data
        current_result = {
            "profile": h_history[::50],
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "t_lin": t_lin[::50],
            "r_lin": r_lin,
            "z_lin": z_lin,
            "Temp": evap_params.T,
            "hmax": params.hmax0,
        }
        results[i] = current_result
    torch.save(results, "data/mdm_sim_poly_1.pth")


if __name__ == "__main__":
    main()
