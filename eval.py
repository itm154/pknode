import argparse
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
from tqdm import tqdm

import utils
from data import PKData
from model import PKNODE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, help="The config file", default="config.toml"
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots for all patients"
    )
    args = parser.parse_args()

    config = utils.getConfig(args.config)

    # Model initialization, same used in training
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

    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"Using {device} device")

    model_path = os.path.join("./models", f"{config['model']['name']}.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        exit()

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    cov_means = checkpoint.get("cov_means")
    cov_stds = checkpoint.get("cov_stds")

    model.eval()

    data = PKData(
        config["data"]["test_file"],
        config["data"]["columns"],
        cov_means=cov_means,
        cov_stds=cov_stds,
    )

    MSE = MSELoss()
    all_losses = []
    all_preds = []
    all_target = []
    all_times = []

    plots_dir = os.path.join(".", "plots")
    if args.save_plots:
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir)
        os.makedirs(plots_dir, exist_ok=True)

    print(f"Evaluating {len(data.patients)} patients...")

    for patient in tqdm(data.patients):
        p_data = data.get_patient_data(patient)
        p_times = p_data["times"]
        p_admin_times = p_data["admin_times"]

        with torch.no_grad():
            pred = utils.solve_multi_dose_ode(data, patient, model, include_cov)

            # Filter out observations before first dose if any
            mask = p_times > p_admin_times[0]
            target_vals = p_data["conc"][mask]

            target = torch.tensor(
                target_vals,
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)

            if target.shape[0] > 0:
                log_pred = torch.log(torch.clamp(pred, min=0) + 1e-7)
                log_target = torch.log(torch.clamp(target, min=0) + 1e-7)
                loss = MSE(log_pred, log_target)

                all_losses.append(loss.item())
                all_preds.extend(pred.cpu().numpy().flatten())
                all_target.extend(target.cpu().numpy().flatten())
                all_times.extend(p_times[mask])

        if args.save_plots:
            p_times_plot = list(p_data["times"]) + list(p_data["admin_times"])
            t_start = min(p_times_plot)
            t_end = max(p_times_plot)

            t_eval = torch.linspace(t_start, t_end, 500, device=device)

            with torch.no_grad():
                t, sol = utils.solve_multi_dose_ode_at_t(
                    data, patient, model, include_cov, t_eval
                )

            plt.figure(figsize=(10, 6))
            plt.plot(t.cpu(), sol.cpu(), label="PKNODE Prediction")
            plt.scatter(
                p_data["times"],
                p_data["conc"],
                color="red",
                label="Ground Truth",
                zorder=5,
            )
            for admin_time in p_data["admin_times"]:
                plt.axvline(x=admin_time, color="gray", linestyle="--", alpha=0.5)

            plt.xlabel("Time")
            plt.ylabel("Concentration (mg/l)")
            plt.title(f"Patient Drug Concentration Prediction for ID: {patient}")
            plt.legend()

            plots_path = os.path.join("./plots", f"{patient}.png")
            plt.savefig(plots_path)
            plt.close()

    if all_losses:
        avg_msle = sum(all_losses) / len(all_losses)
        print(f"\nAverage MSLE across all patients: {avg_msle:.4e}")

        preds_arr = np.array(all_preds)
        target_arr = np.array(all_target)
        times_arr = np.array(all_times)
        residuals = target_arr - preds_arr

        # Residual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Residuals vs Time
        ax1.scatter(times_arr, residuals, alpha=0.5)
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("Time (h)")
        ax1.set_ylabel("Residual (Observed - Predicted)")
        ax1.set_title("Residuals vs Time")

        # Observed vs Predicted plot
        ax2.scatter(preds_arr, target_arr, alpha=0.5)
        lims = [
            np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()]),
        ]
        ax2.plot(lims, lims, "r--", alpha=0.75, zorder=0)
        ax2.set_xlabel("Predicted Concentration")
        ax2.set_ylabel("Observed Concentration")
        ax2.set_title("Observed vs Predicted")

        plt.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)
        residual_plot_path = os.path.join(plots_dir, "residuals.png")
        plt.savefig(residual_plot_path)
        print(f"Residual plots saved to {residual_plot_path}")
        plt.close()
    else:
        print("\nNo valid observations found for evaluation.")
