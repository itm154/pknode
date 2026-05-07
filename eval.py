# pyright: reportPrivateImportUsage=false
import os

import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint

import utils
from data import PKData
from model import PKNODE

if __name__ == "__main__":
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

    # Todo...
