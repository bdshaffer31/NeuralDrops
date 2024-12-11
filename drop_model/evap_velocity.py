import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.integrate as integrate

@dataclass
class FieldVariables:
    u_grid: np.ndarray  # Radial velocity grid
    w_grid: np.ndarray  # Vertical velocity grid
    p_grid: np.ndarray  # Pressure grid
    eta_grid: np.ndarray  # Viscosity grid
    sigma_grid: np.ndarray  # Surface tension grid
    rho_grid: np.ndarray  # Density grid
    diff_grid: np.ndarray  # Diffusivity grid

    m_dot_grid: np.ndarray  # Diffusivity grid


@dataclass
class SimulationParams:
    r_c: float  # Radius of the droplet in meters
    hmax0: float  # Initial droplet height at the center in meters
    Nr: int  # Number of radial points
    Nz: int  # Number of z-axis points
    Nt: int  # Number of time steps
    dr: float  # Radial grid spacing
    dz: float  # Vertical grid spacing
    dt: float  # Time step size
    rho: float  # Density of the liquid (kg/m^3)
    w_e: float  # Constant evaporation rate (m/s)
    sigma: float  # Surface tension (N/m)
    eta: float  # Viscosity (Pa*s)
    d_sigma_dr: float  # Surface tension gradient

    # Antoine's Equation
    A: float  
    B: float
    C: float

    D: float #Diffusvity of Vapor
    Mw: float #Molecular weight of Vapor

    Rs: float #Gas Constant
    T: float #Temperature of drop exterior
    RH: float #Relative Humidity

def setup_grids(params: SimulationParams):
    """Set up the grid arrays and initial field values."""
    # Radial and vertical grids
    r = np.linspace(params.dr, params.r_c, params.Nr)  # r grid (avoiding r=0)
    z = np.linspace(0, params.hmax0, params.Nz)  # z grid

    # Initialize field arrays
    field_vars = FieldVariables(
        u_grid=np.zeros((params.Nr, params.Nz)),  # r velocity
        w_grid=np.zeros((params.Nr, params.Nz)),  # z velocity
        p_grid=np.zeros((params.Nr)),  # pressure
        eta_grid=params.eta * np.ones((params.Nr, params.Nz)),  # constant viscosity
        sigma_grid=params.sigma * np.ones((params.Nr)),  # constant surface tension
        rho_grid=params.rho * np.ones((params.Nr, params.Nz)),  # density
        diff_grid=params.rho * np.ones((params.Nr, params.Nz)),  # diffusivity

        m_dot_grid=np.zeros((params.Nr)),  # mass loss
    )

    return r, z, field_vars

def as_grad(x, dx):
    """Axis symmetric gradient (left side neumann boundary condition)"""
    x_padded = np.pad(x, (1, 0), mode="edge")
    grad_x = np.gradient(x_padded, dx, edge_order=2)
    return grad_x[1:]

def setup_cap_initial_h_profile(r, h0, r_c):
    R = (r_c**2+h0**2)/(2*h0)
    h = np.sqrt((2.0*R*(r+R)-np.square(r+R))) - (R - h0)
    return h

def mass_loss (r,theta):
    p_sat = 10**(params.A-params.B/(params.C+params.T-273.15)) #Antoine's Equation
    p_sat = p_sat/760.0*101325.0 # conversion (mmHg) to (Pa)
    #print(p_sat)

    R_c = params.r_c #TODO
    b_c = 1.0 #TODO
    J_delta = 0 #Contribution from nonhomogenous surface concentration (multi-phase drops)

    cosh_alpha = (r**2*np.cos(theta) + R_c*np.sqrt(np.square(R_c)-np.square(r)*np.square(np.sin(theta))))/(np.square(R_c)-np.square(r))

    #def integral(cosh_alpha):
        #alpha = jnp.arccosh(cosh_alpha)
        #integral = integrate.quad(lambda x:np.tanh(np.pi*x/(2*np.pi*theta))/np.cosh(np.pi*x/(2*np.pi*theta))/np.sqrt(np.cosh(x)-cosh_alpha) , alpha , np.inf)
        #return integral
    integral = np.zeros_like(field_vars.m_dot_grid)
    for i in range(len(r)):
        integral[i] , err = integrate.quad(lambda x:
                                           np.tanh(np.pi*x/(2.0*np.pi-2.0*theta))/
                                           (np.cosh(np.pi*x/(2.0*np.pi-2.0*theta))*np.sqrt(np.cosh(x)-cosh_alpha[i]))
                                           , np.arccosh(cosh_alpha[i]) , np.inf)

    #integral_array = vmap(integral)(cosh_alpha)
    N_prime_alpha = np.sqrt(2)*np.power(np.sqrt((cosh_alpha+np.cos(theta))),3)
    J_c = np.pi*N_prime_alpha*integral/(2*np.sqrt(2)*np.square(np.pi-theta))
    J_c[-1]=J_c[-2]*2

    J_term = (b_c - params.RH)*J_c + J_delta

    m_dot = params.D*params.Mw*p_sat/(params.Rs*params.T*R_c)*J_term
    return m_dot


def evap_velocity (m_dot,h):
    dh_dr = as_grad(h,params.dr)
    w_e = -m_dot/params.rho* np.sqrt(1 + np.square(dh_dr))
    return w_e

# Define the simulation parameters
params = SimulationParams(
    r_c=1e-3,          # Radius of the droplet in meters
    hmax0=5e-4,        # Initial droplet height at the center in meters
    Nr=200,            # Number of radial points
    Nz=110,            # Number of z-axis points
    Nt=2,              # Number of time steps
    dr=1e-3 / 200,     # Radial grid spacing
    dz=5e-4 / 110,     # Vertical grid spacing
    dt=1e-5,           # Time step size eg 1e-5
    rho=1,             # Density of the liquid (kg/m^3) eg 1
    w_e=-0.0,          # Constant evaporation rate (m/s) eg 1e-4
    sigma=0.072,       # Surface tension (N/m) eg 0.072
    eta=1e-3,          # Viscosity (Pa*s) eg 1e-3
    d_sigma_dr=0.0,    # Surface tension gradient

    A = 8.07131,
    B = 1730.63,
    C = 233.4,
    D = 2.42e-5,
    Mw = 0.018,
    Rs = 8.314, # Gas Constant (J/(K*mol))
    T = 293.15,
    RH = 0.20,
)

# Initialize the grids and field variables
r, z, field_vars = setup_grids(params)
h = setup_cap_initial_h_profile(r, params.hmax0, params.r_c)

m_dot = mass_loss(r,20.0*np.pi/180.0)
print(m_dot)
plt.plot(r,m_dot)
plt.show()

w_e = evap_velocity(m_dot,h)

plt.plot(r,w_e)
plt.show()

res = list(map(lambda i: i < 0.6 * params.hmax0, h)).index(True)

print ('The index of element just greater than 0.6 : ' + str(res))

import numpy as np

a = np.array([2, 23, 15, 7, 9, 11, 17, 19, 5, 3])
print(a)

a[3:] = 0
print(a)