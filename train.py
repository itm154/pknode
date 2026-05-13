# pyright: reportPrivateImportUsage=false
# pyright: reportPossiblyUnboundVariable=false

import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    patience: int = 5,
    factor: float = 0.5,
    model_name: str = "model",
    resume: bool = False,
):
    # Find out which device the model is located in so we can use it for other things
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience
    )

    start_epoch = 0
    checkpoint_path = os.path.join("./models", f"{model_name}_checkpoint.pth")
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    MSE = nn.MSELoss()
    losses = []

    accumulation_steps = 4  # Accumulate gradients over 4 patients

    # Train for N epochs, each epoch we go through M patients
    for epoch in range(start_epoch, epochs):
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

            if target.numel() > 0:
                log_pred = torch.log(torch.clamp(pred, min=0) + 1e-7)
                log_target = torch.log(torch.clamp(target, min=0) + 1e-7)
                loss = MSE(log_pred, log_target)

                # Scale loss for gradient accumulation
                (loss / accumulation_steps).backward()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                pbar.set_postfix(
                    {
                        "loss": f"{loss_val:.2e}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(data.patients):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.extend(epoch_losses)
            print(f"Epoch {epoch + 1} finished - Avg Loss: {avg_loss:.4e}")
            scheduler.step(avg_loss)

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_path,
        )

    # Remove checkpoint after successful training
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="The config file to use", default="config.toml"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--load-model", type=str, help="Load an existing model to fine-tune"
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

    if args.load_model:
        if os.path.exists(args.load_model):
            print(f"Loading existing model from {args.load_model}")
            checkpoint = torch.load(
                args.load_model, map_location=device, weights_only=False
            )
            # Handle both full checkpoints and simple state dicts
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Model file {args.load_model} not found.")

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
        patience=train_settings.get("patience", 5),
        factor=train_settings.get("factor", 0.5),
        model_name=config["model"]["name"],
        resume=args.resume,
    )

    # Save model
    utils.save_model(
        model,
        config["model"]["name"],
        "./models",
        cov_means=data.cov_means if hasattr(data, "cov_means") else None,  # pyright: ignore
        cov_stds=data.cov_stds if hasattr(data, "cov_stds") else None,  # pyright: ignore
    )
