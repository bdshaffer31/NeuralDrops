import torch

class PureDropModel:
    def __init__(self, params, evap_model=None, smoothing_fn=None):
        # Initialize with a height profile and a params object
        self.params = params
        self.r, self.z = self.setup_grids()
        self.evap_model = evap_model
        self.smoothing_fn = smoothing_fn

    def setup_grids(self):
        r = torch.linspace(-self.params.r_grid, self.params.r_grid, self.params.Nr)
        z = torch.linspace(0, self.params.hmax0, self.params.Nz)
        return r, z

    @staticmethod
    def grad(x, dx):
        return torch.gradient(x, spacing=dx, edge_order=2)[0]

    @staticmethod
    def safe_inv(x, epsilon=0.0):
        return 1 / (x + epsilon)

    # Curvature calculation
    def calc_curvature(self, h):
        dh_dr = self.grad(h, self.params.dr)
        curvature_term = (self.r * dh_dr) / torch.sqrt(1 + dh_dr**2)
        return curvature_term

    # Pressure calculation
    def calc_pressure(self, h):
        curvature_term = self.calc_curvature(h)
        d_curvature_dr = self.grad(curvature_term, self.params.dr)
        pressure = -self.params.sigma * self.safe_inv(self.r, 0.0) * d_curvature_dr

        h_star = self.params.hmax0/100
        n = 3
        m = 2
        theta_e = 2*torch.arctan(torch.tensor(self.params.hmax0/(0.5*self.params.r_grid)))
        dis_press = -self.params.sigma*torch.square(torch.tensor(theta_e))*(n-1)*(m-1)/(n-m)/(2*h_star)*(torch.pow(torch.tensor(h_star/self.params.hmax0), n)-torch.pow(torch.tensor(h_star/self.params.hmax0), m))
        return pressure #+ dis_press

    # u velocity calculation
    def calc_u_velocity(self, h):
        u_grid = torch.zeros(
            (self.params.Nr, self.params.Nz), device=h.device
        )  # r velocity
        pressure = self.calc_pressure(h)
        dp_dr = self.grad(pressure, self.params.dr)

        # Broadcasting for vectorized computation
        h_r = h.unsqueeze(1)
        z_grid = self.z.unsqueeze(0)
        dp_dr = dp_dr.unsqueeze(1)

        # Compute integrand and integrate along z dimension
        integrand = -(dp_dr * (h_r - z_grid)) / self.params.eta
        u_grid = torch.cumsum(
            (integrand[:, :-1] + integrand[:, 1:]) * 0.5 * self.params.dz, dim=1
        )
        u_grid = torch.cat(
            [torch.zeros((self.params.Nr, 1), device=h.device), u_grid], dim=1
        )

        u_grid = self.interp_h_mask_grid(u_grid, h, self.z)

        u_grid[u_grid > 5] = 5
        u_grid[u_grid < -5] = -5
        return u_grid

    # w velocity calculation
    def calc_w_velocity(self, h, u_grid):
        w_grid = torch.zeros((self.params.Nr, self.params.Nz), device=h.device)
        ur_grid = u_grid * self.r.unsqueeze(1)
        d_ur_grid_dr = torch.gradient(ur_grid, spacing=self.params.dr, dim=0)[0]
        div_r = self.safe_inv(self.r).unsqueeze(1)  # Shape: (Nr, 1)
        integrand = div_r * d_ur_grid_dr
        w_grid[:, 1:] = -torch.cumsum(
            (integrand[:, :-1] + integrand[:, 1:]) * 0.5 * self.params.dz, dim=1
        )
        w_grid = self.interp_h_mask_grid(w_grid, h, self.z)

        w_grid[w_grid > 10] = 10
        w_grid[w_grid < -10] = -10
        return w_grid

    def interp_h_mask_grid(self, grid_data, h, z):
        dz = z[1] - z[0]
        masked_grid = grid_data.clone()

        lower_indices = torch.searchsorted(z, h) - 1
        valid_mask = (lower_indices >= 0) & (lower_indices < len(z) - 1)

        z_below = z[lower_indices[valid_mask]]
        h_valid = h[valid_mask]
        occupation_percent = (h_valid - z_below) / dz

        masked_grid[valid_mask, lower_indices[valid_mask] + 1] *= occupation_percent
        for i, idx in zip(
            torch.nonzero(valid_mask, as_tuple=True)[0], lower_indices[valid_mask]
        ):
            masked_grid[i, idx + 2 :] = 0
        invalid_mask = ~valid_mask
        masked_grid[invalid_mask] = 0

        return masked_grid

    def apply_smoothing_fn(self, x):
        if self.smoothing_fn is None:
            return x
        return self.smoothing_fn(x)

    # Flow-induced dh/dt calculation
    def calc_flow_dh_dt(self, h):
        u_grid = self.calc_u_velocity(h)

        # Integrate r * u over z from 0 to h for each radial position
        integral_u_r = torch.trapezoid(
            self.r.unsqueeze(1) * u_grid, dx=self.params.dz, dim=1
        )
        grad_u_r = self.grad(integral_u_r, self.params.dr) * self.params.dz
        grad_u_r = self.apply_smoothing_fn(grad_u_r)
        flow_dh_dt = -self.safe_inv(self.r) * grad_u_r

        return flow_dh_dt

    # Evaporation-induced dh/dt calculation
    def calc_evap_dh_dt(self, r, h):
        if self.evap_model is None:
            return torch.zeros_like(h)
        return self.evap_model(self.params, r, h)

    # Total dh/dt calculation
    def calc_dh_dt(self, h):
        #print(self.calc_evap_dh_dt(self.r, h))
        return self.calc_flow_dh_dt(h) + self.calc_evap_dh_dt(self.r, h)


def main():
    import utils_old as utils_old
    import drop_viz as drop_viz
    import evap_models

    drop_viz.set_styling()
    torch.set_default_dtype(torch.float64)

    #from recycling_bin.load_data_old import ProfileDataset

    #dataset = ProfileDataset(
    #    "data", [40], axis_symmetric=False, spatial_subsample=6, temporal_subsample=24
    #)
    #viz_file = dataset.valid_files[0]
    #h_0 = dataset.data[viz_file]["profile"][1]
    #h_0 = dataset.profile_scaler.inverse_apply(h_0)
    #print(torch.max(h_0), torch.min(h_0))
    #h_0 -= torch.min(h_0)
    #h_0 -= 0.6
    #h_0 = torch.clamp(h_0, min=0.0)
    #h_0 = h_0.to(torch.float64)
    #h_0 = utils.drop_polynomial_fit(h_0, 8)

    #h_0 *= 0.000003 * 100
    #r_c = 0.000003 * 640
    #maxh0 = torch.max(h_0).item() * 1.2
    #print(h_0.shape, maxh0, print(r_c))

    # TODO consider doing something different with these
    # params = utils.SimulationParams(
    #     r_c=r_c,  # Radius of the droplet in meters
    #     hmax0=maxh0,  # Initial droplet height at the center in meters
    #     Nr=214,  # Number of radial points
    #     Nz=110,  # Number of z-axis points
    #     dr=2 * r_c / (214 - 1),  # Radial grid spacing
    #     dz=maxh0 / (110 - 1),  # Vertical grid spacing
    #     rho=1,  # Density of the liquid (kg/m^3) eg 1
    #     sigma=0.072,  # Surface tension (N/m) eg 0.072
    #     eta=1e-3,  # Viscosity (Pa*s) eg 1e-3
    # )
    params = utils_old.SimulationParams(
        r_grid=1.0e-3,  # Radius of the droplet in meters
        hmax0=3e-4,  # Initial droplet height at the center in meters
        Nr=640,  # Number of radial points
        Nz=110,  # Number of z-axis points
        dr= 2 * 1.0e-3 / (640 - 1),  # Radial grid spacing
        dz=3e-4 / (110 - 1),  # Vertical grid spacing
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-3,  # Viscosity (Pa*s) eg 1e-5

        A = 8.07131, # Antoine Equation (-)
        B = 1730.63, # Antoine Equation (-)
        C = 233.4, # Antoine Equation (-)
        D = 2.42e-5, # Diffusivity of H2O in Air (m^2/s)
        Mw = 0.018, # Molecular weight H2O vapor (kg/mol)
        #Rs = 8.314, # Gas Constant (J/(K*mol))
        Rs = 461.5, # Gas Constant (J/(K*kg))
        T = 293.15, # Ambient Temperature (K)
        RH = 0.20, # Relative Humidity (-)
    )
    Nt = 3000
    dt = 1e-2
    t_lin = torch.linspace(0, dt * Nt, Nt)

    def smoothing_fn(x):
        return utils_old.gaussian_blur_1d(x, sigma=10)

    drop_model = PureDropModel(params, evap_model=evap_models.deegan_evap_model, smoothing_fn=smoothing_fn)

    r_c = 0.9*params.r_grid

    #h_0 = utils.setup_polynomial_initial_h_profile(
    #    drop_model.r, 0.8 * params.hmax0, r_c, order=4
    #)

    h_0 = utils_old.setup_cap_initial_h_profile(drop_model.r, 0.8 * params.hmax0, r_c
    )

    drop_viz.flow_viz(drop_model, h_0, 0, 0)

    def post_fn(h):
        h = torch.clamp(h, min=0)  # ensure non-negative height
        h = utils_old.drop_polynomial_fit(h, 8)  # project height on polynomial basis
        return h
    
    #print(h_0)

    h_history = utils_old.run_forward_euler_simulation(drop_model, h_0, t_lin, post_fn)
    drop_viz.plot_height_profile_evolution(drop_model.r, h_history, params)

    # plot the velocity profile and
    # drop_viz.inspect(drop_model, h_history[-1].clone())
    drop_viz.plot_velocity(drop_model, h_history[-1].clone())
    #drop_viz.inspect(drop_model, h_history[0].clone())
    #drop_viz.plot_velocity(drop_model, h_history[0].clone(), 0, 0)
    drop_viz.flow_viz(drop_model, h_history[-1].clone(), 0, 0)
    #drop_viz.flow_viz(drop_model, h_history[-1].clone())


if __name__ == "__main__":
    main()
