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
        activation_fn=nn.ReLU(),
        output_fn=nn.Identity(),
    ):
        super(FCNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(output_fn)
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


class FNO(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_fno_layers=4,
        modes=16,
        width=64,
        activation_fn=nn.ReLU(),
        num_fc_layers=2,
        fc_width=128,
    ):
        """
        Generalized Fourier Neural Operator.

        Args:
            input_dim (int): Number of input features per grid point.
            output_dim (int): Number of output features per grid point.
            num_fno_layers (int): Number of Fourier layers.
            modes (int): Number of Fourier modes to retain.
            width (int): Number of channels in Fourier layers.
            activation_fn (torch.nn.Module): Activation function.
            fc_width (int): Size of the first fully connected hidden layer.
            num_fc_layers (int): Number of fully connected layers after Fourier layers.
        """
        super(FNO, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_fno_layers = num_fno_layers
        self.modes = modes
        self.width = width
        self.fc_width = fc_width
        self.activation_fn = activation_fn
        self.num_fc_layers = num_fc_layers

        # Input projection layer
        self.fc0 = nn.Linear(self.input_dim, self.width)

        # Fourier layers
        self.fourier_layers = nn.ModuleList(
            [
                SpectralConv1d(self.width, self.width, self.modes)
                for _ in range(self.num_fno_layers)
            ]
        )
        self.w = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_fno_layers)]
        )

        # Fully connected layers after Fourier layers
        fc_layers = [nn.Linear(self.width, self.fc_width), self.activation_fn]
        for _ in range(self.num_fc_layers - 2):
            fc_layers.append(nn.Linear(self.fc_width, self.fc_width))
            fc_layers.append(self.activation_fn)
        fc_layers.append(nn.Linear(self.fc_width, self.output_dim))
        self.final_fc = nn.Sequential(*fc_layers)

    def forward(self, h_0, z, t):
        batch_size, grid_size = h_0.shape

        # Expand z and t to match the spatial dimensions of h_0
        z_expanded = z.unsqueeze(1).expand(
            batch_size, grid_size, -1
        )  # [batch, grid_size, num_conditions]
        t_expanded = t.unsqueeze(1).expand(
            batch_size, grid_size, -1
        )  # [batch, grid_size, 1]

        # Expand h_0 to include a feature dimension
        h_0_expanded = h_0.unsqueeze(-1)  # [batch, grid_size, 1]

        # Concatenate inputs along the feature dimension
        input_data = torch.cat(
            [h_0_expanded, z_expanded, t_expanded], dim=-1
        )  # [batch, grid_size, input_dim]

        # Apply the first fully connected layer
        x = self.fc0(input_data)  # [batch, grid_size, width]
        x = x.permute(0, 2, 1)  # [batch, width, grid_size]

        # Apply Fourier layers
        for fourier_layer, w_layer in zip(self.fourier_layers, self.w):
            x1 = fourier_layer(x)
            x2 = w_layer(x)
            x = x1 + x2
            x = self.activation_fn(x)

        x = x.permute(0, 2, 1)  # Back to [batch, grid_size, width]

        # Apply final fully connected layers
        x = self.final_fc(x)  # [batch, grid_size, output_dim]

        if self.output_dim == 1:
            return x.squeeze(-1)  # Final output: [batch, grid_size]
        return x  # Final output: [batch, grid_size, output_dim]


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # Complex multiplication
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # Perform FFT
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        # Multiply relevant modes
        out_ft = torch.zeros_like(x_ft, dtype=torch.cfloat)
        out_ft[:, :, : self.modes] = self.compl_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )

        # Perform Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


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
        "sine": Sine(),  # Adding the custom sine activation
    }
    activation = activations.get(activation_name.lower())
    if activation is None:
        raise ValueError(f"Activation function '{activation_name}' is not supported.")
    return activation
