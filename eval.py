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
    parser.add_argument("-p", type=int, help="The patient ID", required=True)
    args = parser.parse_args()

    config = utils.getConfig()

    # Model initialization
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

    # Load weights
    model_settings = config["model"]
    model_path = os.path.join(model_settings["path"], f"{model_settings['name']}.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Please train the model first.")
        exit()

    weights = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(weights)
    model.eval()

    data = PKData(config["data"]["test_file"], config["data"]["columns"])
    p_data = data.get_patient_data(args.p)

    p_times = list(p_data["times"]) + list(p_data["admin_times"])
    t_start = min(p_times)
    t_end = max(p_times)

    # Add a small buffer (e.g., 5%) to the end for better visualization
    t_end = t_end * 1.05

    t_eval = torch.linspace(t_start, t_end, 150, device=device)

    with torch.no_grad():
        t, sol = utils.solve_multi_dose_ode_at_t(
            data, args.p, model, include_cov, t_eval
        )

    plt.figure(figsize=(10, 6))
    plt.scatter(
        p_data["times"], p_data["conc"], color="red", label="Ground Truth", zorder=5
    )

    # Model prediction as a line
    plt.plot(t.cpu(), sol.cpu(), label="PKNODE Prediction", color="blue")
    plt.plot(t.cpu(), sol.cpu())
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.title(f"Patient Drug Concentration Prediction for ID: {args.p}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
