from torchdiffeq import odeint_adjoint as odeint
from torch import nn
import torch


class FCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=128,
        num_hidden_layers=4,
        activation_fn=nn.ReLU,
    ):
        super(FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.model(x)


class ODE_FCNN(FCNN):
    def __init__(self, input_dim, output_dim, conditioning_dim, *args, **kwargs):
        input_nodes = input_dim + conditioning_dim
        super(ODE_FCNN, self).__init__(input_nodes, output_dim, *args, **kwargs)
        self.conditioning = None

    def set_conditioning(self, z):
        self.conditioning = z

    def forward(self, t, x):
        input_data = torch.cat([x, self.conditioning.expand(x.size(0), -1)], dim=1)
        return self.model(input_data)


class NeuralODE(nn.Module):
    def __init__(self, ode_func, solver="rk4"):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.solver = solver

    def forward(self, x0, z, t):
        self.ode_func.set_conditioning(z)
        return odeint(self.ode_func, x0, t, method=self.solver)


def get_activation(activation_name):
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "softmax": nn.Softmax(dim=1),
        "identity": nn.Identity(),
    }
    activation = activations.get(activation_name.lower())
    if activation is None:
        raise ValueError(f"Activation function '{activation_name}' is not supported.")
    return activation
