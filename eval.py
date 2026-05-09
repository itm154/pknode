# pyright: reportPrivateImportUsage=false
import argparse
import os

import matplotlib.pyplot as plt
import torch

import utils
from data import PKData
from model import PKNODE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--patient", type=int, help="The patient ID", required=True
    )
    parser.add_argument(
        "-c", "--config", type=str, help="The config file", default="config.toml"
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
    p_data = data.get_patient_data(args.patient)

    p_times = list(p_data["times"]) + list(p_data["admin_times"])
    t_start = min(p_times)
    t_end = max(p_times)

    t_eval = torch.linspace(t_start, t_end, 500, device=device)

    with torch.no_grad():
        t, sol = utils.solve_multi_dose_ode_at_t(
            data, args.patient, model, include_cov, t_eval
        )

    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu(), sol.cpu(), label="PKNODE Prediction")
    plt.scatter(p_data["times"], p_data["conc"], color="red", label="Ground Truth")
    for admin_time in p_data["admin_times"]:
        plt.axvline(x=admin_time, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"Patient Drug Concentration Prediction for ID: {args.patient}")
    plt.legend()

    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    plots_path = os.path.join("./plots", f"{args.patient}.png")
    print(f"Plot for Patient ID {args.patient} saved to {plots_path}")
    plt.savefig(plots_path)
