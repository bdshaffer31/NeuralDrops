import numpy as np
from scipy.ndimage import gaussian_filter

# TODO remove
import torch


class PureDropModel:
    def __init__(self, params, evap_model=None, sigma=10):
        # initialize with a height profile and a params object
        # setup grids
        self.params = params
        self.r, self.z = self.setup_grids()
        self.evap_model = evap_model
        self.sigma = 10

    def setup_grids(self):
        r = np.linspace(
            -self.params.r_c, self.params.r_c, self.params.Nr
        )  # r grid (avoiding r=0)
        z = np.linspace(0, self.params.hmax0, self.params.Nz)  # z grid
        return r, z

    @staticmethod
    def grad(x, dx):
        return np.gradient(x, dx, edge_order=2)

    @staticmethod
    def safe_inv(x, epsilon=0.0):
        return 1 / (x + epsilon)

    # pressure
    def calc_pressure(self, h):
        curvature_term = self.calc_curvature(h)
        d_curvature_dr = self.grad(curvature_term, self.params.dr)
        pressure = -self.params.sigma * self.safe_inv(self.r) * d_curvature_dr
        return pressure

    # curvature
    def calc_curvature(self, h):
        dh_dr = self.grad(h, self.params.dr)
        curvature_term = (self.r * dh_dr) / np.sqrt(1 + dh_dr**2)
        return curvature_term

    # u velocity
    def calc_u_velocity(self, h):
        u_grid = np.zeros((self.params.Nr, self.params.Nz))  # r velocity
        pressure = self.calc_pressure(h)
        dp_dr = self.grad(pressure, self.params.dr)
        for i in range(len(self.r)):
            h_r = h[i]
            for j, z_val in enumerate(self.z):
                integrand = -(dp_dr[i]) * (h_r - self.z) / self.params.eta
                u_grid[i, j] = np.trapz(integrand[: j + 1], dx=self.params.dz)
        u_grid = self.interp_h_mask_grid(u_grid, h, self.z)

        return u_grid

    # w velocty
    def calc_w_velocity(self, h, u_grid):
        w_grid = np.zeros((self.params.Nr, self.params.Nz))
        ur_grid = u_grid * self.r[:, None]
        d_ur_grid_dr = np.gradient(ur_grid, self.params.dr, axis=0)
        div_r = self.safe_inv(self.r)
        integrand = div_r[:, None] * d_ur_grid_dr
        for j in range(1, len(self.z)):  # ignore BC
            w_grid[:, j] = -1 * np.trapz(
                integrand[:, : j + 1], dx=self.params.dz, axis=1
            )
        w_grid = self.interp_h_mask_grid(w_grid, h, self.z)
        w_grid[0, :] = w_grid[1, :]  # hack to deal with numerical issues at r ~ 0
        return w_grid

    def interp_h_mask_grid(self, grid_data, h, z):
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

    # flow_dh_dt
    def calc_flow_dh_dt(self, h):
        u_grid = self.calc_u_velocity(h)

        # Integrate r * u over z from 0 to h for each radial position
        integral_u_r = np.trapz(self.r[:, None] * u_grid, dx=self.params.dz, axis=1)
        integral_u_r *= self.params.dz
        grad_u_r = np.gradient(integral_u_r, self.params.dr)

        # blur to smooth
        grad_u_r = gaussian_filter(grad_u_r, sigma=self.sigma)
        flow_dh_dt = -1 * self.safe_inv(self.r) * grad_u_r
        return flow_dh_dt

    # evap_dh_dt
    def calc_evap_dh_dt(self, h):
        if self.evap_model is None:
            return np.zeros_like(h)
        return self.evap_model(h)

    # calc_dh_dt
    # def calc_dh_dt(self, h):
    #     return self.calc_flow_dh_dt(h) + self.calc_evap_dh_dt(h)

    def calc_dh_dt(self, h):
        return self.calc_flow_dh_dt(h) + self.calc_evap_dh_dt(h)


def main():
    import drop_model.utils as utils
    import drop_model.drop_viz_numpy as drop_viz
    import torch

    params = utils.SimulationParams(
        r_c=1e-3,  # Radius of the droplet in meters
        hmax0=3e-4,  # Initial droplet height at the center in meters
        Nr=204,  # Number of radial points
        Nz=110,  # Number of z-axis points
        dr=2 * 1e-3 / (204 - 1),  # Radial grid spacing
        dz=3e-4 / (110 - 1),  # Vertical grid spacing
        rho=1,  # Density of the liquid (kg/m^3) eg 1
        sigma=0.072,  # Surface tension (N/m) eg 0.072
        eta=1e-5,  # Viscosity (Pa*s) eg 1e-3
    )
    Nt = 10
    dt = 1e-3
    t_lin = np.linspace(0, dt * Nt, Nt)

    def evap_model(h, kappa=0.0):
        return -kappa * np.ones_like(h)

    drop_model = PureDropModel(params, evap_model=evap_model, sigma=5)

    h_0 = utils.setup_parabolic_initial_h_profile(
        torch.tensor(drop_model.r), params.hmax0, params.r_c, order=4
    ).numpy()

    # h_history = utils.run_forward_euler_simulation(drop_model, h_0, t_lin)
    h_history = utils.run_forward_euler_simulation_numpy(drop_model, h_0, t_lin)
    drop_viz.plot_height_profile_evolution(drop_model.r, h_history, params)

    # plot the velocity profile and
    drop_viz.inspect(drop_model, h_history[-1].copy())
    drop_viz.plot_velocity(drop_model, h_history[-1].copy())
    drop_viz.inspect(drop_model, h_history[0].copy())
    drop_viz.plot_velocity(drop_model, h_history[0].copy())


if __name__ == "__main__":
    main()
