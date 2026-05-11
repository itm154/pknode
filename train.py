# pyright: reportPrivateImportUsage=false
# pyright: reportPossiblyUnboundVariable=false

import argparse

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    use_scheduler = True if step_size and gamma is not None else False
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size, gamma)  # pyright: ignore

    MSE = nn.MSELoss()
    losses = []

    accumulation_steps = 4  # Accumulate gradients over 4 patients

    # Train for N epochs, each epoch we go through M patients
    for epoch in range(epochs):
        pbar = tqdm(data.patients, desc=f"Epoch {epoch + 1}/{epochs}")
        epoch_losses = []
        optimizer.zero_grad()

        for i, patient in enumerate(pbar):
            pred = utils.solve_multi_dose_ode(data, patient, model, include_cov)

            p_data = data.get_patient_data(patient)
            p_times = p_data["times"]
            p_admin_times = p_data["admin_times"]

            mask = p_times > p_admin_times[0]
            target_vals = p_data["conc"][mask]

            target = torch.tensor(
                target_vals,
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)

            loss = MSE(pred, target)

            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data.patients):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * accumulation_steps
            epoch_losses.append(loss_val)
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.2e}",
                    "lr": f"{scheduler.get_last_lr()[0] if use_scheduler else learning_rate:.2e}",
                }
            )

        if use_scheduler:
            scheduler.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.extend(epoch_losses)
        print(f"Epoch {epoch + 1} finished - Avg Loss: {avg_loss:.4e}")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="The config file to use", default="config.toml"
    )
    args = parser.parse_args()

    config = utils.getConfig(args.config)

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
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
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
    train(
        model,
        data,
        epochs=train_settings["epoch"],
        learning_rate=train_settings["learning_rate"],
        weight_decay=train_settings["weight_decay"],
        include_cov=include_cov,
        step_size=train_settings.get("step_size"),
        gamma=train_settings.get("gamma"),
    )

    # Save model
    utils.save_model(
        model,
        config["model"]["name"],
        "./models",
        cov_means=data.cov_means if hasattr(data, "cov_means") else None,  # pyright: ignore
        cov_stds=data.cov_stds if hasattr(data, "cov_stds") else None,  # pyright: ignore
    )
