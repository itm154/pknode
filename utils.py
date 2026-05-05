# pyright: reportPrivateImportUsage=false
# pyright: reportArgumentType=false

import tomllib
import os

import torch
from torchdiffeq import odeint

from data import PKData
from model import PKNODE


def getConfig() -> dict:
    """
    Get configuration from config.toml
    """
    try:
        with open("config.toml", "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def solve_multi_dose_ode(data: PKData, patient, node: PKNODE, include_cov: bool):
    # Determine the device from the model
    device = next(node.parameters()).device

    p_data = data.get_patient_data(patient)
    times = torch.tensor(p_data["times"], dtype=torch.float32, device=device)
    admin_times = torch.tensor(
        p_data["admin_times"], dtype=torch.float32, device=device
    )
    doses = torch.tensor(p_data["doses"], dtype=torch.float32, device=device)

    if include_cov:
        node.v = torch.tensor(p_data["covariates"], dtype=torch.float32, device=device)
        V = node.net_V(node.v)
    else:
        V = node.V_param

    sol_tot = torch.tensor([], device=device)
    last_conc = torch.tensor([0.0], device=device)

    for j in range(len(admin_times)):
        t_start = admin_times[j]
        t_end = admin_times[j + 1] if j < len(admin_times) - 1 else times[-1] + 0.01

        mask = (times >= t_start) & (times < t_end)
        t_obs = times[mask]

        t_vector = torch.cat([t_start.unsqueeze(0), t_obs])
        if j < len(admin_times) - 1:
            t_vector = torch.cat([t_vector, t_end.unsqueeze(0)])

        dose = doses[j]
        node.z0 = (dose / V).view(1)
        node.n_admin = torch.tensor([j + 1.0], device=device)

        conc_init = last_conc + node.z0  # pyright: ignore

        sol = odeint(
            node,
            conc_init,
            t_vector - t_vector[0],
            method="dopri5",
            atol=1e-3,  # Speed up training by loosening tolerance
            rtol=1e-3,
        )

        if j < len(admin_times) - 1:
            sol_tot = torch.cat([sol_tot, sol[1:-1]])
            last_conc = sol[-1]
        else:
            sol_tot = torch.cat([sol_tot, sol[1:]])

    return sol_tot


def save_model(model: PKNODE, name: str, path: str):
    full_path = os.path.join(path, f"{name}.pth")
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")
