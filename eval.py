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
        "-c",
        "--config",
        type=str,
        help="Path to the TOML configuration file",
        default="config.toml",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save concentration-time profiles and residual analysis plots",
    )
    args = parser.parse_args()

    config = utils.getConfig(args.config)

    # Model initialization, same used in training
    nn_settings = config["settings"]["nn"]
    model_settings = config["model"]
    absorption = model_settings.get("absorption", False)

    dim_c = nn_settings["dim_c"]
    if nn_settings["include_covariates"]:
        dim_V = nn_settings["dim_V"]
        n_cov = len(config["data"]["columns"]["covariates"])
        model = PKNODE(dim_c, dim_V, n_cov, absorption)
        include_cov = True
    else:
        model = PKNODE(dim_c, absorption=absorption)
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

    model.eval()

    data = PKData(
        config["data"]["test_file"],
        config["data"]["columns"],
    )

    all_preds = []
    all_target = []
    all_times = []

    plots_dir = os.path.join(".", "plots")

    print(f"Evaluating {len(data.patients)} patients...")

    patient_results = []
    for patient in tqdm(data.patients, desc="Analyzing data"):
        p_data = data.get_patient_data(patient)
        p_times = p_data["times"]
        p_admin_times = p_data["admin_times"]

        with torch.no_grad():
            res = utils.solve_dose_ode(data, patient, model, include_cov)
            if isinstance(res, tuple):
                _, pred = res
            else:
                pred = res

            # Extract central compartment concentration
            central_idx = 1 if model.absorption else 0
            if pred.numel() > 0:
                pred = pred[:, central_idx].view(-1, 1)

            # Filter out observations before first dose if any
            mask = p_times > p_admin_times[0]
            target_vals = p_data["conc"][mask]

            target = torch.tensor(
                target_vals,
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)

            if target.shape[0] > 0:
                all_preds.extend(pred.cpu().numpy().flatten())
                all_target.extend(target.cpu().numpy().flatten())
                all_times.extend(p_times[mask])

        patient_results.append((patient, p_data))

    if all_target:
        preds_arr = np.array(all_preds)
        target_arr = np.array(all_target)
        times_arr = np.array(all_times)

        # Log scale metrics for band calculation
        log_preds = np.log(np.maximum(preds_arr, 1e-7))
        log_target = np.log(np.maximum(target_arr, 1e-7))
        log_residuals = log_target - log_preds
        sigma_log = np.std(log_residuals)
        z_score = 1.645  # 90% CI for prediction interval

        # Linear scale metrics
        mse_lin = np.mean((preds_arr - target_arr) ** 2)
        rmse_lin = np.sqrt(mse_lin)
        mae_lin = np.mean(np.abs(preds_arr - target_arr))
        mape = np.mean(np.abs((target_arr - preds_arr) / (target_arr + 1e-7))) * 100

        # R-squared (Linear)
        ss_res = np.sum((target_arr - preds_arr) ** 2)
        ss_tot = np.sum((target_arr - np.mean(target_arr)) ** 2)
        r2_lin = 1 - (ss_res / (ss_tot + 1e-7))

        # R-squared (Log)
        ss_res_log = np.sum(log_residuals**2)
        ss_tot_log = np.sum((log_target - np.mean(log_target)) ** 2)
        r2_log = 1 - (ss_res_log / (ss_tot_log + 1e-7))

        # MSLE
        msle = np.mean(log_residuals**2)

        print(f"\nEvaluation Results for {config['model']['name']}:")
        print(f"{'Metric':<20} | {'Value':<10}")
        print("-" * 35)
        print(f"{'MAE (Linear)':<20} | {mae_lin:.4e}")
        print(f"{'MAPE (%)':<20} | {mape:.2f}%")
        print(f"{'RMSE (Linear)':<20} | {rmse_lin:.4e}")
        print(f"{'MSLE (Log)':<20} | {msle:.4e}")
        print(f"{'R-squared (Linear)':<20} | {r2_lin:.4f}")
        print(f"{'R-squared (Log)':<20} | {r2_log:.4f}")

        if args.save_plots:
            if os.path.exists(plots_dir):
                shutil.rmtree(plots_dir)
            os.makedirs(plots_dir, exist_ok=True)

            for patient, p_data in tqdm(patient_results, desc="Generating plots"):
                p_times_plot = list(p_data["times"]) + list(p_data["admin_times"])
                t_start = min(p_times_plot)
                t_end = max(p_times_plot)
                t_eval = torch.linspace(t_start, t_end, 500, device=device)

                with torch.no_grad():
                    t, sol = utils.solve_dose_ode(
                        data, patient, model, include_cov, t_eval=t_eval
                    )
                    central_idx = 1 if model.absorption else 0
                    sol = sol[:, central_idx].view(-1, 1)

                pred_curve = sol.cpu().numpy().flatten()
                # 90% Prediction Interval band (log-normal assumption)
                lower_bound = pred_curve * np.exp(-z_score * sigma_log)
                upper_bound = pred_curve * np.exp(z_score * sigma_log)

                plt.figure(figsize=(10, 6))
                plt.plot(t.cpu(), pred_curve, label="PKNODE Prediction", color="blue")
                plt.fill_between(
                    t.cpu(),
                    lower_bound,
                    upper_bound,
                    color="blue",
                    alpha=0.2,
                    label="90% Interval",
                )

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
                plt.ylabel("Concentration")
                plt.title(f"Patient Drug Concentration Prediction for ID: {patient}")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plots_path = os.path.join(plots_dir, f"{patient}.png")
                plt.savefig(plots_path, dpi=300)
                plt.close()

            # Residual plots
            residuals = target_arr - preds_arr
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            ax1.scatter(times_arr, residuals, alpha=0.5)
            ax1.axhline(y=0, color="r", linestyle="--")
            ax1.set_xlabel("Time (h)")
            ax1.set_ylabel("Residual (Observed - Predicted)")
            ax1.set_title("Residuals vs Time")

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
            residual_plot_path = os.path.join(plots_dir, "residuals.png")
            plt.savefig(residual_plot_path)
            print(f"Plots and residuals saved to {plots_dir}")
            plt.close()
    else:
        print("\nNo valid observations found for evaluation.")
