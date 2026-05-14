# pyright: reportPrivateImportUsage=false
# pyright: reportArgumentType=false

import os
import tomllib

import torch
from torch import Tensor
from torchdiffeq import odeint

from data import PKData
from model import PKNODE


def getConfig(file: str) -> dict:
    """
    Get configuration from config.toml
    """
    try:
        with open(file, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def solve_multi_dose_ode(
    data: PKData,
    patient,
    node: PKNODE,
    include_cov: bool,
):
    """
    Solve the ODE for a patient using the Neural Network model
    """
    # Determine which device the model is using
    device = next(node.parameters()).device

    # Get the specific patient's data
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

        mask = (times > t_start) & (times <= t_end)
        t_obs = times[mask]

        t_vector = torch.cat([t_start.unsqueeze(0), t_obs])

        # Ensure there is atleast two points for integration
        if j < len(admin_times) - 1 or len(t_obs) == 0:
            if t_end > t_start:
                t_vector = torch.cat([t_vector, t_end.unsqueeze(0)])

        dose = doses[j]
        node.z0 = (dose / V).view(1)  # pyright: ignore
        node.n_admin = torch.tensor([j + 1.0], device=device)

        conc_init = last_conc + node.z0  # pyright: ignore

        sol = odeint(
            node,
            conc_init,
            t_vector - t_vector[0],
            method="dopri5",
        )

        if j < len(admin_times) - 1:
            sol_tot = torch.cat([sol_tot, sol[1:-1]])
            last_conc = sol[-1]
        else:
            sol_tot = torch.cat([sol_tot, sol[1:]])

    return sol_tot


def solve_multi_dose_ode_at_t(
    data: PKData,
    patient_id: int,
    model: PKNODE,
    include_cov: bool,
    t_eval: Tensor,
):
    """
    Solve the ODE at time t_eval
    """
    device = next(model.parameters()).device
    p_data = data.get_patient_data(patient_id)
    admin_times = torch.tensor(
        p_data["admin_times"], dtype=torch.float32, device=device
    )
    doses = torch.tensor(p_data["doses"], dtype=torch.float32, device=device)

    if include_cov:
        model.v = torch.tensor(p_data["covariates"], dtype=torch.float32, device=device)
        V = model.net_V(model.v)
    else:
        V = model.V_param

    last_conc = torch.tensor([0.0], device=device)
    all_t = []
    all_sol = []

    for j in range(len(admin_times)):
        t_start = admin_times[j]
        t_end = admin_times[j + 1] if j < len(admin_times) - 1 else t_eval[-1]

        mask = (t_eval >= t_start) & (t_eval <= t_end)
        t_interval = t_eval[mask]

        if len(t_interval) == 0:
            t_vec = torch.tensor([t_start, t_end], device=device)
        else:
            t_vec = torch.cat([t_start.unsqueeze(0), t_interval])
            if t_vec[-1] < t_end:
                t_vec = torch.cat([t_vec, t_end.unsqueeze(0)])

        if len(t_vec) > 1 and t_vec[1] == t_vec[0]:
            t_vec = torch.cat([t_vec[0:1], t_vec[2:]])

        dose = doses[j]
        model.z0 = (dose / V).view(1)  # pyright:ignore
        model.n_admin = torch.tensor([j + 1.0], device=device)
        conc_init = last_conc + model.z0  # pyright: ignore

        sol = odeint(
            model,
            conc_init,
            t_vec - t_vec[0],
            method="dopri5",
            atol=1e-8,
            rtol=1e-8,
        )

        if len(t_interval) > 0:
            start_idx = 0 if t_interval[0] == t_start else 1
            end_idx = start_idx + len(t_interval)
            all_sol.append(sol[start_idx:end_idx])
            all_t.append(t_interval)

        last_conc = sol[-1]

    return torch.cat(all_t), torch.cat(all_sol)


def save_model(
    model: PKNODE,
    name: str,
    path: str,
    cov_means: list | None = None,
    cov_stds: list | None = None,
):
    """
    Saves a model with a specified name and path, including scaling parameters
    """
    full_path = os.path.join(path, f"{name}.pth")
    os.makedirs(path, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "cov_means": cov_means,
        "cov_stds": cov_stds,
    }
    torch.save(state, full_path)
    print(f"Model saved to {full_path}")
