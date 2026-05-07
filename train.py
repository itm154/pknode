# pyright: reportPrivateImportUsage=false
# pyright: reportPossiblyUnboundVariable=false

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

import utils
from data import PKData
from model import PKNODE


def train(
    model: PKNODE,
    data: PKData,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    include_cov: bool = False,
    step_size: int | None = None,
    gamma: float | None = None,
):
    # Find out which device the model is located in so we can use it for other things
    device = next(model.parameters()).device

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    use_scheduler = True if step_size and gamma is not None else False
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size, gamma)  # pyright: ignore

    MSE = nn.MSELoss()
    losses = []

    # Train for N epochs, each epoch we go through M patients
    # Everything here is largely unchanged from pharmacoNODE
    # other than changes to how the patient data is handled
    # Read: https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
    for epoch in range(epochs):
        pbar = tqdm(data.patients, desc=f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = []
        for patient in pbar:
            optimizer.zero_grad()

            pred = utils.solve_multi_dose_ode(data, patient, model, include_cov)

            p_data = data.get_patient_data(patient)
            p_times = p_data["times"]
            p_admin_times = p_data["admin_times"]

            # Target only observations after the first dose
            # to match the behavior of solve_multi_dose_ode
            mask = p_times > p_admin_times[0]
            target_vals = p_data["conc"][mask]

            target = torch.tensor(
                target_vals,
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)

            loss = MSE(pred, target)
            loss.backward()

            # Gradient clipping to prevent gradient/losses going crazy
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            optimizer.step()
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.2e}",
                    "lr": f"{scheduler.get_last_lr() if use_scheduler else learning_rate}",
                }
            )

        if use_scheduler:
            pbar.set_postfix()
            scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.extend(epoch_losses)
        print(f"Epoch {epoch + 1} finished - Avg Loss: {avg_loss:.4e}")

    return losses


if __name__ == "__main__":
    config = utils.getConfig()

    # Model initialization
    # Get configuration from config.toml file
    nn_settings = config["settings"]["nn"]
    dim_c = nn_settings["dim_c"]
    if nn_settings["include_covariates"]:
        dim_V = nn_settings["dim_V"]
        n_cov = len(config["data"]["columns"]["covariates"])
        model = PKNODE(dim_c, dim_V, n_cov)
        include_cov = True
    else:
        model = PKNODE(dim_c)
        include_cov = False

    # Tru to use GPU/CUDA if available
    device = (
        torch.accelerator.current_accelerator().type  # pyright: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.to(device)  # Move model to device

    print(model)
    print(f"Using {device} device")

    # Initialize data (use config.toml to define dataset)
    data = PKData(config["data"]["train_file"], config["data"]["columns"])

    # Train model
    train_settings = config["settings"]["train"]
    if train_settings["use_scheduler"]:
        train(
            model,
            data,
            epochs=train_settings["epoch"],
            learning_rate=train_settings["learning_rate"],
            weight_decay=train_settings["weight_decay"],
            include_cov=include_cov,
            step_size=train_settings["step_size"],
            gamma=train_settings["gamma"],
        )
    else:
        train(
            model,
            data,
            epochs=train_settings["epoch"],
            learning_rate=train_settings["learning_rate"],
            weight_decay=train_settings["weight_decay"],
            include_cov=include_cov,
        )

    # Save model
    model_settings = config["model"]
    utils.save_model(model, model_settings["name"], model_settings["path"])
