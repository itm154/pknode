# pyright: reportPrivateImportUsage=false
import shutil
import argparse
import os

import matplotlib.pyplot as plt
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
        torch.accelerator.current_accelerator().type  # pyright: ignore
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

    if args.save_plots and os.path.exists("./plots"):
        shutil.rmtree("./plots")
        os.makedirs("./plots")
    elif args.save_plots and not os.path.exists("./plots"):
        os.makedirs("./plots")

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
                loss = MSE(pred, target)

                all_losses.append(loss.item())

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
        avg_mse = sum(all_losses) / len(all_losses)
        print(f"\nAverage MSE across all patients: {avg_mse:.4e}")

    else:
        print("\nNo valid observations found for evaluation.")
