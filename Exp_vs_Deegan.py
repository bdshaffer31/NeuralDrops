import torch
import pure_drop_model
import drop_model.evap_models as evap_model

def main():
    import drop_model.utils as utils
    import drop_model.drop_viz as drop_viz

    drop_viz.set_styling()
    torch.set_default_dtype(torch.float64)

    from load_data import ProfileDataset

    dataset = ProfileDataset(
        "data", [42], axis_symmetric=False, spatial_subsample=5, temporal_subsample=24
    )
    viz_file = dataset.valid_files[0]
    np_profile = dataset["profile"]
    profile = torch.tensor(np_profile, dtype=torch.float32)
    h_0 = dataset.data[viz_file]["profile"][0]
    h_0 = dataset.profile_scaler.inverse_apply(h_0)
    #print(torch.max(h_0), torch.min(h_0))
    h_0 -= torch.min(h_0)
    h_0 /= torch.max(h_0)
    h_0 -= 0.6
    h_0 = torch.max(torch.zeros_like(h_0), h_0)

    import matplotlib.pyplot as plt

    plt.plot(h_0)
    print(h_0.dtype)
    h_0 = h_0.to(torch.float64)
    h_0 = utils.drop_polynomial_fit(h_0, 8)
    plt.plot(h_0)
    plt.show()
    h_0 *= 0.001
    r_c = 0.000003 * 640
    maxh0 = torch.max(h_0).item() * 1.2

    # TODO consider doing something different with these
    params = utils.SimulationParams(
        r_c=r_c,  # Radius of the droplet in meters
        hmax0=maxh0,  # Initial droplet height at the center in meters
        Nr=256,  # Number of radial points
        Nz=110,  # Number of z-axis points
        dr= 2 * r_c / (256 - 1),  # Radial grid spacing
        dz=maxh0 / (110 - 1),  # Vertical grid spacing
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-3,  # Viscosity (Pa*s) eg 1e-3

        A = 8.07131, # Antoine Equation (-)
        B = 1730.63, # Antoine Equation (-)
        C = 233.4, # Antoine Equation (-)
        D = 2.42e-5, # Diffusivity of H2O in Air (m^2/s)
        Mw = 0.018, # Molecular weight H2O vapor (kg/mol)
        Rs = 8.314, # Gas Constant (J/(K*mol))
        T = 293.15, # Ambient Temperature (K)
        RH = 0.20, # Relative Humidity (-)
    )

    Nt = 800
    dt = 1e-3
    t_lin = torch.linspace(0, dt * Nt, Nt)

    def smoothing_fn(x):
        return utils.gaussian_blur_1d(x, sigma=10)

    drop_model = pure_drop_model.PureDropModel(params, evap_model=evap_model.deegan_evap_model, smoothing_fn=smoothing_fn)

    #h_0 = utils.setup_parabolic_initial_h_profile(
    #    drop_model.r, 0.8 * params.hmax0, params.r_c, order=4
    #)

    drop_viz.flow_viz(drop_model, h_0)

    def post_fn(h):
        h = torch.clamp(h, min=0)
        h = utils.drop_polynomial_fit(h, 8)
        return h

    h_history = utils.run_forward_euler_simulation(drop_model, h_0, t_lin, post_fn)
    drop_viz.plot_height_profile_evolution(drop_model.r, h_history, params)

    drop_viz.plot_height_profile_evolution(drop_model.r, torch.tensor(dataset.data[viz_file]["profile"][0]), params)

if __name__ == "__main__":
    main()