import os
import tomllib
from typing import Any, Optional

import torch
from torch import Tensor
from torchdiffeq import odeint

from data import PKData
from model import PKNODE


def getConfig(file: str) -> dict:
    """
    Returns dictionary from configuration file
    """
    try:
        with open(file, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}


def solve_dose_ode(
    data: PKData,
    patient: Any,
    model: PKNODE,
    include_cov: bool,
    t_eval: Optional[Tensor] = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Solve the ODE for a patient using the Neural Network model
    """
    device = next(model.parameters()).device
    p_data = data.get_patient_data(patient)

    t0 = torch.tensor(p_data["admin_times"][0], dtype=torch.float32, device=device)
    dose = torch.tensor(p_data["doses"][0], dtype=torch.float32, device=device)

    if include_cov:
        model.v = torch.tensor(p_data["covariates"], dtype=torch.float32, device=device)
        V = model.predict_V(model.v)
    else:
        V = model.V_param

    V = V if V is not None else torch.tensor([1.0], device=device)
    z0 = torch.zeros(model.state_dim, device=device)
    z0[0] = dose / V
    model.z0 = z0

    if t_eval is None:
        times = torch.tensor(p_data["times"], dtype=torch.float32, device=device)
        mask = times > t0
        t_target = times[mask]
    else:
        mask = t_eval >= t0
        t_target = t_eval[mask]

    if len(t_target) == 0:
        return (
            (torch.tensor([], device=device), torch.tensor([], device=device))
            if t_eval is not None
            else torch.tensor([], device=device)
        )

    # Prepare time vector for integration (must start at 0 for t0)
    t_vec = torch.cat([t0.unsqueeze(0), t_target])

    # Remove duplicate if t_target[0] == t0
    if len(t_target) > 0 and t_target[0] == t0:
        t_vec = t_vec[1:]
        sol = odeint(model, model.z0, t_vec - t0, method="dopri5")
    else:
        sol = odeint(model, model.z0, t_vec - t0, method="dopri5")
        sol = sol[1:]

    return (t_target, sol) if t_eval is not None else sol


def save_checkpoint(state: dict, name: str, path: str):
    """
    Save a training checkpoint
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{name}_checkpoint.pth")
    torch.save(state, full_path)


def save_model(model: PKNODE, name: str, path: str):
    """
    Save the final model
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{name}.pth")
    state = {"model_state_dict": model.state_dict()}
    torch.save(state, full_path)
    print(f"Model saved to {full_path}")
