import torch
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks
import logger
import load_data


@torch.no_grad
def validate(model_type, model, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    if model_type in ["node", "flux_fno", "fno_node"]:
        for t, x_0, z, y in val_loader:
            y_pred = model(x_0, z, t[0]).transpose(0, 1)
            loss = loss_fn(y_pred, y)
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
                # print("t", t.shape, torch.min(t), torch.max(t))
                # print("x_0", x_0.shape, torch.min(x_0), torch.max(x_0))
                # print("z", z.shape, torch.min(z), torch.max(z))
                # print("y_traj", y_traj.shape, torch.min(y_traj), torch.max(y_traj))
                optimizer.zero_grad()
                pred_y_traj = model(x_0, z, t[0]).transpose(0, 1)
                loss = loss_fn(pred_y_traj, y_traj)
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
    
    from drop_model import pure_drop_model, utils
    def smoothing_fn(x):
        return utils.gaussian_blur_1d(x, sigma=10)
    

    #TODO Possibly grad grid params from data
    # params = utils.SimulationParams(
    #     r_grid = 1280 * model_config["pixel_resolution"],
    #     hmax0=1024 * model_config["pixel_resolution"] * model_config["profile_scale"],
    #     Nr=int(1280 / model_config["spatial_sampling"])+1,
    #     Nz=110,
    #     dr= 2 * 1280 * model_config["pixel_resolution"] / (int(1280 / model_config["spatial_sampling"]) - 1),
    #     dz=1024 * model_config["pixel_resolution"] * model_config["profile_scale"] / (110 - 1),
    #     rho=1,
    #     sigma=0.072,
    #     eta=1e-3,

    #     #TODO 
    #     A = 8.07131,
    #     B = 1730.63,
    #     C = 233.4,
    #     D = 2.42e-5,
    #     Mw = 0.018,
    #     #Rs = 8.314,
    #     Rs = 461.5,
    #     T = 293.15,
    #     RH = 0.20,
    #     )

    data = torch.load(config["data_file"])
    first_key = list(data.keys())[0]
    t_lin = data[first_key]["t_lin"]
    r_lin = data[first_key]["r_lin"]
    z_lin = data[first_key]["z_lin"]
    dt = t_lin[1] - t_lin[0]
    dr = r_lin[1] - r_lin[0]
    dz = z_lin[1] - z_lin[0]

    params = utils.SimulationParams(
        r_grid = torch.max(r_lin),
        hmax0=torch.max(z_lin),
        Nr=r_lin.shape[0],
        Nz=z_lin.shape[0],
        dr=dr,
        dz=dz,
        rho=1,
        sigma=0.072,
        eta=1e-3,

        #TODO 
        A = 8.07131,
        B = 1730.63,
        C = 233.4,
        D = 2.42e-5,
        Mw = 0.018,
        #Rs = 8.314,
        Rs = 461.5,
        T = 293.15,
        RH = 0.20,
        )

    flow_model = pure_drop_model.PureDropModel(params, smoothing_fn=smoothing_fn)

    print("dt", dt)
    # TODO random ass dt scaling to get shit moving
    dt = dt * 1
    ode_func = networks.FNOFluxODEWrapper(fno_model, flow_model, profile_scale=config["profile_scale"], time_scale=dt)
    model = networks.FNOFluxODESolver(ode_func, time_step=dt, solver_type=model_config["solver"] )
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
