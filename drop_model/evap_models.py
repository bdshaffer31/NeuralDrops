import torch
import numpy as np
import scipy.integrate as integrate


def no_evap_model(params, h, z=None, kappa=0.0e-3):
    return -kappa * torch.ones_like(h)


def constant_evap_model(params, h, z=None, kappa=1.0e-7):
    return -kappa * torch.ones_like(h)

# TODO make this a class, init with r and params, __call__ only needs h
def deegan_evap_model(evap_params, params, r, h):
    def grad(x, dx):
        return torch.gradient(x, spacing=dx, edge_order=2)[0]

    def mass_loss(evap_params, r_in, theta, R_c):
        p_sat = 10 ** (
            evap_params.A - evap_params.B / (evap_params.C + evap_params.T - 273.15)
        )  # Antoine's Equation
        p_sat = p_sat / 760.0 * 101325.0  # conversion (mmHg) to (Pa)
        # print(p_sat)

        # R_c = params.r_c #TODO
        b_c = 1.0  # TODO
        J_delta = 0  # Contribution from nonhomogenous surface concentration (multi-phase drops)

        lam = -(torch.pi - 2.0 * theta) / (2.0 * torch.pi - 2.0 * theta)
        J_c = 0.6 * torch.pow(1 - torch.square(r_in / R_c), lam)

        J_c[-1] = J_c[-2] * 2
        J_c[0] = J_c[1] * 2

        J_term = (b_c - evap_params.RH) * J_c + J_delta

        m_dot = evap_params.D * evap_params.Mw * p_sat / (evap_params.Rs * evap_params.T * R_c) * J_term
        #import matplotlib.pyplot as plt
        #plt.plot(r_in,m_dot)
        return m_dot

    m_dot = torch.zeros_like(r)
    h_max_current = torch.max(h)

    # mask = torch.abs(h) > 0.009 * params.hmax0
    # non_zero_indices = torch.where(mask)[0]
    # start_idx = non_zero_indices[0]  # + 1
    # end_idx = non_zero_indices[-1] + 1
    # r_c_current = (r[end_idx] - r[start_idx]) / 2

    # r_flux = torch.linspace(
    #        -r_c_current, r_c_current, (end_idx - start_idx)
    #    )  # r grid (avoiding r=0)

    num_c = list(map(lambda i: i > 0.01 * params.hmax0, h)).index(True)
    #mask = torch.abs(h) > 1.0e-2 * params.hmax0
    #non_zero_indices = torch.where(mask)[0]
    #num_c = non_zero_indices[0]
    #num_c = int(num_c)
    r_c_current = -r[num_c]
    theta_current = 2 * torch.arctan(h_max_current / r_c_current)

    # m_dot[start_idx:end_idx] = mass_loss(params, r_flux, theta_current, r_c_current)
    m_dot[num_c:-(num_c)] = mass_loss(
        evap_params, r[num_c:-(num_c)], theta_current, r_c_current
    )

    dh_dr = grad(h, params.dr)
    w_e = -m_dot / params.rho * torch.sqrt(1 + torch.square(dh_dr))
    return w_e - 2.0e-7


def diddens_evap_model(params, r, h):
    def grad(x, dx):
        return torch.gradient(x, spacing=dx, edge_order=2)[0]

    def mass_loss(params, r_in, theta, R_c):
        p_sat = 10 ** (
            params.A - params.B / (params.C + params.T - 273.15)
        )  # Antoine's Equation
        p_sat = p_sat / 760.0 * 101325.0  # conversion (mmHg) to (Pa)
        # print(p_sat)

        # R_c = params.r_c #TODO
        b_c = 1.0  # TODO
        J_delta = 0  # Contribution from nonhomogenous surface concentration (multi-phase drops)

        cosh_alpha = np.zeros_like(r_in)

        cosh_alpha[1:-1] = (
            r_in[1:-1] ** 2 * np.cos(theta)
            + R_c
            * np.sqrt(np.square(R_c) - np.square(r_in[1:-1]) * np.square(np.sin(theta)))
        ) / (np.square(R_c) - np.square(r_in[1:-1]))

        integral = np.zeros_like(r_in)
        for i in range(1, len(r_in) - 1):
            integral[i], err = integrate.quad(
                lambda x: np.tanh(np.pi * x / (2.0 * np.pi - 2.0 * theta))
                / (
                    np.cosh(np.pi * x / (2.0 * np.pi - 2.0 * theta))
                    * np.sqrt(np.cosh(x) - cosh_alpha[i])
                ),
                np.arccosh(cosh_alpha[i]),
                np.inf,
            )

        N_prime_alpha = np.sqrt(2) * np.power(np.sqrt((cosh_alpha + np.cos(theta))), 3)
        J_c = (
            np.pi
            * N_prime_alpha
            * integral
            / (2 * np.sqrt(2) * np.square(np.pi - theta))
        )
        # J_c[-1]=J_c[-2]*2
        # J_c[0]=J_c[1]*2

        J_c[-1] = J_c[-2]
        J_c[0] = J_c[1]

        J_term = (b_c - params.RH) * J_c + J_delta

        m_dot = params.D * params.Mw * p_sat / (params.Rs * params.T * R_c) * J_term
        return m_dot

    h_max_current = np.max(h)
    num_c = list(map(lambda i: i > 0.01 * params.hmax0, h)).index(True)
    r_c_current = -r[num_c]
    theta_current = 2 * np.arctan(h_max_current / r_c_current)

    m_dot = np.zeros_like(r)
    m_dot[num_c:-num_c] = mass_loss(params, r[num_c:-num_c], theta_current, r_c_current)

    dh_dr = grad(h, params.dr)
    w_e = -m_dot / params.rho * np.sqrt(1 + np.square(dh_dr))
    return w_e
