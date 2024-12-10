import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import networks
import visualize
import logger
import load_data


def validate_fno_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for h_0, z, t, h_t in val_loader:
            pred_h_t = model(h_0, z, t)
            loss = loss_fn(pred_h_t, h_t)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_fno(
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

        for h_0, z, t, h_t in train_loader:
            optimizer.zero_grad()
            pred_h_t = model(h_0, z, t)
            loss = loss_fn(pred_h_t, h_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = validate_fno_model(model, val_loader, loss_fn)

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
    activation_fn = networks.get_activation(config["activation_fn"])

    # Load the best model from the logger
    best_model_path = log_loader.get_relpath("best_model.pth")
    model = networks.FNO(
        config["input_dim"],
        config["output_dim"],
        num_fno_layers=config["num_fno_layers"],
        modes=config["modes"],
        width=config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=config["num_fc_layers"],
        fc_width=config["fc_width"],
    )

    # Load the best validation model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model


def run_training(config, run_dir):
    exp_logger = logger.ExperimentLogger(run_dir=run_dir, use_timestamp=False)

    # Load dataset
    # data = load_data.setup_fno_data(
    #     batch_size=config["batch_size"],
    #     exp_nums=config["exp_nums"],
    #     valid_solutes=config["valid_solutes"],
    #     valid_substrates=config["valid_substrates"],
    #     valid_temps=config["valid_temps"],
    #     temporal_subsample=config["temporal_subsample"],
    #     spatial_subsample=config["spatial_subsample"],
    #     use_log_transform=config["use_log_transform"],
    #     data_dir=config["data_dir"],
    #     test_split=config["val_ratio"],
    # )
    data = load_data.setup_data(config)
    train_loader, val_loader, profile_data = data

    # Initialize FNO model, loss function, and optimizer
    initial_condition, conditioning, t, target_snapshot = next(iter(train_loader))
    grid_size = initial_condition.shape[1]
    conditioning_dim = conditioning.shape[1]
    input_dim = conditioning_dim + 2  # dim z + t + h_0 value
    output_dim = 1

    config["grid_size"] = grid_size
    config["conditioning_dim"] = conditioning_dim
    config["input_dim"] = input_dim
    config["output_dim"] = output_dim
    activation_fn = networks.get_activation(config["activation_fn"])

    model = networks.FNO(
        input_dim,
        output_dim,
        num_fno_layers=config["num_fno_layers"],
        modes=config["modes"],
        width=config["fno_width"],
        activation_fn=activation_fn,
        num_fc_layers=config["num_fc_layers"],
        fc_width=config["fc_width"],
    )

    exp_logger.log_config(config)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train the FNO model
    train_fno(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        config["num_epochs"],
        exp_logger,
    )


def main(train=False):
    config = {
        # Training params
        "manual_seed": 42,
        "num_epochs": 2,
        "lr": 1e-2,
        # Model params
        "model_type": "fno",
        "modes": 16,
        "num_fno_layers": 4,
        "fno_width": 256,
        "num_fc_layers": 4,
        "fc_width": 256,
        "activation_fn": "relu",
        # Data params
        "data_dir": "data",
        "batch_size": 32,
        "exp_nums": [1],  # [10, 15, 18, 9, 6, 8, 48, 47], # None = use all experiments
        "valid_solutes": None,  # None = keep all solutes
        "valid_substrates": None,  # None = keep all substrates
        "valid_temps": None,  # None = keep all temperatures
        "temporal_subsample": 15,  # Temporal subsampling of profile data
        "spatial_subsample": 5,
        "use_log_transform": False,
        "val_ratio": 0.1,
    }
    torch.manual_seed(config["manual_seed"])

    run_dir = "test_fno_operator"
    if train:
        run_training(config, run_dir)
    visualize.viz_results(run_dir)


if __name__ == "__main__":
    main(train=True)
