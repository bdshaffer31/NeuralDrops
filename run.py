import torch
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks
import logger
import load_data

import drop_model.utils as drop_utils
from drop_model import pure_drop_model


@torch.no_grad
def validate(model_type, model, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    if model_type in ["node", "flux_fno", "fno_node"]:
        for t, x_0, z, y in val_loader:
            t = t[0]
            sub_step = 9
            total_points = (len(t) - 1) * (sub_step + 1) + 1
            t_sub_stepped = torch.linspace(t[0], t[-1], steps=total_points)

            y_pred = model(x_0, z, t_sub_stepped).transpose(0, 1)
            y_pred_original_t = y_pred[:, :: sub_step + 1]
            loss = loss_fn(y_pred_original_t, y)
            val_loss += loss.item()
    elif model_type == "fno":
        for t, h_0, z, h_t in val_loader:
            pred_h_t = model(h_0, z, t)
            loss = loss_fn(pred_h_t, h_t)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train(
    model_type,
    model,
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

        if model_type in ["node", "fno_node", "flux_fno"]:
            for t, x_0, z, y_traj in train_loader:
                optimizer.zero_grad()

                t = t[0]
                sub_step = 9
                total_points = (len(t) - 1) * (sub_step + 1) + 1
                t_sub_stepped = torch.linspace(t[0], t[-1], steps=total_points)

                pred_y_traj = model(x_0, z, t_sub_stepped).transpose(0, 1)
                pred_y_traj_original_t = pred_y_traj[:, :: sub_step + 1]
                loss = loss_fn(pred_y_traj_original_t, y_traj)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        elif model_type == "fno":
            for t, h_0, z, h_t in train_loader:
                optimizer.zero_grad()
                pred_h_t = model(h_0, z, t)
                loss = loss_fn(pred_h_t, h_t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        val_loss = validate(model_type, model, val_loader, loss_fn)

        logger.log_metrics(
            metrics={"train_mse": avg_train_loss, "val_mse": val_loss}, epoch=epoch
        )
        logger.save_best_model(model, metric=1 / val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4e}", "Val Loss": f"{val_loss:.4e}"}
        )
    logger.save_model(model, file_name="final_model.pth")


def load_model_from_log_loader(log_loader):
    config = log_loader.load_config()

    model = init_model(config)

    best_model_path = log_loader.get_relpath("best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def init_node_model(config):
    model_config = config["model_config"]
    activation_fn = networks.get_activation(model_config["activation_fn"])

    model = networks.ODE_FCNN(
        model_config["input_dim"],
        model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_hidden_layers=model_config["num_hidden_layers"],
        conditioning_dim=model_config["conditioning_dim"],
        activation_fn=activation_fn,
    )
    model = networks.NeuralODE(model, solver=model_config["solver"])
    return model


def init_fno_model(config):
    model_config = config["model_config"]
    activation_fn = networks.get_activation(model_config["activation_fn"])
    model = networks.FNO(
        model_config["input_dim"],
        model_config["output_dim"],
        num_fno_layers=model_config["num_fno_layers"],
        modes=model_config["modes"],
        width=model_config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=model_config["num_fc_layers"],
        fc_width=model_config["fc_width"],
    )
    return model


def init_fno_node(config):
    model_config = config["model_config"]
    activation_fn = networks.get_activation(model_config["activation_fn"])
    fno_model = networks.FNO_Flux(
        model_config["input_dim"],
        model_config["output_dim"],
        num_fno_layers=model_config["num_fno_layers"],
        modes=model_config["modes"],
        width=model_config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=model_config["num_fc_layers"],
        fc_width=model_config["fc_width"],
    )
    ode_func = networks.FNOFluxODEWrapper(fno_model)
    model = networks.FNOFluxODESolver(ode_func, solver_type=model_config["solver"])
    return model


def init_flux_fno(config):
    model_config = config["model_config"]

    # all post-processing function / smoothings regularizations etc for solver
    def flux_post_fn(h):
        h = drop_utils.drop_polynomial_fit_batch(h, 8)
        h = drop_utils.symmetrize(h)
        return h

    def smoothing_fn(x): # reconfigure this into flow post function to be consistent
        return drop_utils.gaussian_blur_1d(x, sigma=10)

    def solver_post_fn(h):
        h = torch.clamp(h, min=0)  # ensure non-negative height
        # h = drop_utils.symmetrize(h) # may be needed
        h = drop_utils.drop_polynomial_fit_batch(
            h, 8
        )  # project height on polynomial basis
        h = drop_utils.symmetrize(h)
        return h

    data = torch.load(config["data_file"])
    drop_params, dt = drop_utils.load_drop_params_from_data(data)

    activation_fn = networks.get_activation(model_config["activation_fn"])
    fno_model = networks.FNO_Flux(
        model_config["input_dim"],
        model_config["output_dim"],
        num_fno_layers=model_config["num_fno_layers"],
        modes=model_config["modes"],
        width=model_config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=model_config["num_fc_layers"],
        fc_width=model_config["fc_width"],
        gamma=model_config["gamma"],
        model_scale=model_config["flux_model_scale"],
        post_fn=flux_post_fn,
    )

    flow_model = pure_drop_model.PureDropModel(drop_params, smoothing_fn=smoothing_fn)
    neural_drop = networks.NeuralDrop(
        fno_model, flow_model, profile_scale=config["profile_scale"], time_scale=dt
    )
    ode_func = networks.FNOFluxODEWrapper(neural_drop)
    model = networks.FNOFluxODESolver(
        ode_func,
        dt=1,
        solver_type=model_config["solver"],
        post_fn=solver_post_fn,
    )
    return model


def init_model(config):
    # model_config = config["model_config"]
    init_fns = {
        "fno": init_fno_model,
        "node": init_node_model,
        "fno_node": init_fno_node,
        "flux_fno": init_flux_fno,
    }
    return init_fns[config["model_type"]](config)


def setup_in_out_dim(config, train_loader):
    model_type = config["model_type"]

    t, h0, z, ht = next(iter(train_loader))

    conditioning_dim = z.shape[1]
    config["model_config"]["conditioning_dim"] = conditioning_dim
    if model_type == "fno":
        config["model_config"]["input_dim"] = conditioning_dim + 2
        config["model_config"]["output_dim"] = 1
    elif model_type == "node":
        config["model_config"]["input_dim"] = h0.shape[1]
        config["model_config"]["output_dim"] = ht.shape[2]
    elif model_type in ["fno_node", "flux_fno"]:
        config["model_config"]["input_dim"] = conditioning_dim + 1
        config["model_config"]["output_dim"] = 1
    return config


def run_training(config, run_dir):
    exp_logger = logger.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    data = load_data.setup_data_from_config(config)
    train_loader, val_loader, profile_data = data

    config = setup_in_out_dim(config, train_loader)
    model = init_model(config)
    exp_logger.log_config(config)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    train(
        config["model_type"],
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        config["num_epochs"],
        exp_logger,
    )
