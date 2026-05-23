import os
import tomllib
from typing import Any, Optional

import torch
from torch import Tensor
from torchdiffeq import odeint

from data import PKData
from model import PKNODE

# Utility functions for common usage
# Improvements from pharmacoNODE:
# 1. Configuration uses toml file, easily parsable and extensible
# 2. Simplified multi dose ODE solver (one function is usable for both plotting and training)
# 3. Models is saved after every epoch in training, enabling resuming or configuration change mid train
# NOTE: A model file must be accompanied by it's configuration file, or else it would not load


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

    admin_times = torch.tensor(
        p_data["admin_times"], dtype=torch.float32, device=device
    )
    doses = torch.tensor(p_data["doses"], dtype=torch.float32, device=device)

    if include_cov:
        model.v = torch.tensor(p_data["covariates"], dtype=torch.float32, device=device)
        V = model.predict_V(model.v)
    else:
        V = model.V_param

    V = V if V is not None else torch.tensor([1.0], device=device)

    # Observation times
    if t_eval is None:
        times = torch.tensor(p_data["times"], dtype=torch.float32, device=device)
    else:
        times = t_eval

    # Filter times to be after or at first dose
    t0 = admin_times[0]
    times = times[times >= t0]

    if len(times) == 0:
        return (
            (torch.tensor([], device=device), torch.tensor([], device=device))
            if t_eval is not None
            else torch.tensor([], device=device)
        )

    # Initial state
    curr_state = torch.zeros(model.state_dim, device=device)
    curr_time = t0

    # Unique dose times
    unique_admin = torch.unique(admin_times).sort()[0]

    # Solutions at requested observation times
    all_sol = []

    for i in range(len(unique_admin)):
        t_dose = unique_admin[i]

        # 1. Integrate from curr_time to t_dose if there's a gap
        if t_dose > curr_time:
            # Check for observations in this gap
            mask = (times >= curr_time) & (times < t_dose)
            t_gap = times[mask]

            t_vec = torch.cat([curr_time.view(1), t_gap, t_dose.view(1)])
            t_vec_unique = torch.unique(t_vec).sort()[0]

            model.z0 = curr_state.clone()
            sol = odeint(model, curr_state, t_vec_unique - t0, method="dopri5")

            # Map solutions back to requested times
            for tj in t_gap:
                idx = (t_vec_unique == tj).nonzero(as_tuple=True)[0][0]
                all_sol.append((tj.item(), sol[idx]))

            curr_state = sol[-1]
            curr_time = t_dose

        # 2. Add dose(s) at this time
        dose_val = doses[admin_times == t_dose].sum()
        curr_state[0] += dose_val / V.item()

    # 3. Final integration for remaining observations
    mask = times >= curr_time
    if mask.any():
        t_rem = times[mask]
        t_vec = torch.cat([curr_time.view(1), t_rem])
        t_vec_unique = torch.unique(t_vec).sort()[0]

        model.z0 = curr_state.clone()
        sol = odeint(model, curr_state, t_vec_unique - t0, method="dopri5")

        for tj in t_rem:
            idx = (t_vec_unique == tj).nonzero(as_tuple=True)[0][0]
            all_sol.append((tj.item(), sol[idx]))

    # Sort results by time to match input 'times'
    all_sol.sort(key=lambda x: x[0])
    sol = torch.stack([x[1] for x in all_sol])

    return (times, sol) if t_eval is not None else sol


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
