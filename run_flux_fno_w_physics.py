import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks_old
import visualize
import logger
import load_data
import utils_old

import pure_drop_model

import drop_model.utils as utils_drop


def validate_node_model(model, time_steps, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for t, x_0, z, y in val_loader:
            evap_pred = model(x_0, z, t[0]).transpose(0, 1)
            drop_model = pure_drop_model.PureDropModel(params, evap_model=evap_pred, smoothing_fn=utils_drop.smoothing_fn)
            y_pred = utils_drop.run_forward_euler_simulation(drop_model, h_0, t_lin, post_fn)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_node(
    model,
    time_steps,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    logger,
):
    pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0

        for t, x_0, z, y_traj in train_loader:
            optimizer.zero_grad()

            evap_pred = model(x_0, z, t[0]).transpose(0, 1)
            drop_model = pure_drop_model.PureDropModel(params, evap_model=evap_pred, smoothing_fn=utils_drop.smoothing_fn)

            
            def post_fn(h):
                h = torch.clamp(h, min=0)
                h = utils_old.drop_polynomial_fit(h, 8)
                return h

            pred_y_traj = utils_drop.run_forward_euler_simulation(drop_model, x_0, t[1], post_fn)

            loss = loss_fn(pred_y_traj, y_traj)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        val_loss = validate_node_model(model, time_steps, val_loader, loss_fn)

        logger.log_metrics(
            metrics={"train_mse": avg_train_loss, "val_mse": val_loss}, epoch=epoch
        )
        logger.save_best_model(model, metric=1 / val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )
    logger.save_model(model, file_name="final_model.pth")


def load_fno_model_from_logger(log_loader):
    config = log_loader.load_config()
    activation_fn = networks_old.get_activation(config["activation_fn"])

    # Setup the FNO Flux model from the config parameters
    fno_model = networks_old.FNO_Flux(
        input_dim=config.get("input_dim"),
        output_dim=config.get("output_dim"),
        num_fno_layers=config.get("num_fno_layers"),
        modes=config.get("modes"),
        width=config.get("fno_width"),
        activation_fn=activation_fn,
        num_fc_layers=config.get("num_fc_layers"),
        fc_width=config.get("fc_width"),
    )
    ode_func = networks_old.FNOFluxODEWrapper(fno_model)
    # model = networks.NeuralODE(ode_func, config.get("solver"))
    # model = networks.ForwardEuler(ode_func)
    model = networks_old.FNOFluxODESolver(ode_func, solver_type=config["solver"])

    # Load the best validation model
    best_model_path = log_loader.get_relpath("best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def run_training(config, run_dir):
    exp_logger = logger.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    data = load_data.setup_data(config)
    train_loader, val_loader, dataset = data

    # Initialize ODE model, loss function, and optimizer
    train_time_steps, initial_condition, conditioning, target_snapshots = next(
        iter(train_loader)
    )
    grid_size = initial_condition.shape[1]
    conditioning_dim = conditioning.shape[1]
    input_dim = 1 + conditioning_dim  # h_0 + conditioning variables
    output_dim = 1

    config["grid_size"] = grid_size
    config["conditioning_dim"] = conditioning_dim
    config["input_dim"] = input_dim
    config["output_dim"] = output_dim
    activation_fn = networks_old.get_activation(config["activation_fn"])

    fno_model = networks_old.FNO_Flux(
        input_dim,
        output_dim,
        num_fno_layers=config["num_fno_layers"],
        modes=config["modes"],
        width=config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=config["num_fc_layers"],
        fc_width=config["fc_width"],
    )
    ode_func = networks_old.FNOFluxODEWrapper(fno_model)
    # model = networks.NeuralODE(ode_func, solver=config["solver"])
    # model = networks.ForwardEuler(ode_func)
    model = networks_old.FNOFluxODESolver(ode_func, solver_type=config["solver"])

    exp_logger.log_config(config)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train the Neural ODE model
    train_node(
        model,
        train_time_steps,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        config["num_epochs"],
        exp_logger,
    )


def main(train=False):
    config = {
        # training params
        "manual_seed": 42,
        "num_epochs": 10,
        "lr": 1e-2,
        # model params
        "model_type": "flux_fno",
        "modes": 16,
        "num_fno_layers": 4,
        "fno_width": 64,
        "num_fc_layers": 4,
        "fc_width": 256,
        "activation_fn": "relu",
        "solver": "euler",
        # data params
        "data_dir": "data",
        "batch_size": 32,
        "exp_nums": utils_old.good_run_numbers()[
            :1
        ],  # if None use all, otherwise give a list of ints
        "valid_solutes": None,  # if None keep all solutes, otherwise give a list of strings
        "valid_substrates": None,  # if None keep all substrates, otherwise give a list of strings
        "valid_temps": None,  # if None keep all substrates, otherwise give a list of floats
        "temporal_subsample": 15,  # temporal subsampling on profile data
        "spatial_subsample": 5,
        "temporal_pad": 128,
        "axis_symmetric": True,  # split along x axis
        "use_log_transform": False,
        "traj_len": 4,
        "val_ratio": 0.1,
    }
    torch.manual_seed(config["manual_seed"])

    run_dir = "run_fno_flux_axis_symmetric"
    if train:
        run_training(config, run_dir)
    visualize.viz_results(run_dir)


if __name__ == "__main__":
    # TODO load config from input + defaults
    # split out train and plotting functions
    main(train=True)
