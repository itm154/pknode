# pyright: reportPrivateImportUsage=false
# pyright: reportArgumentType=false

import torch
from torch import Tensor
import torch.nn as nn


class PKNODE(nn.Module):
    """
    NODE model definition for PK/PD

    z0: Initial concentration
    z(t): Concentration at t
    u: Dose
    v: Covariates (weight, age, etc ...)
    t: time

    dz(t)/dt = f(z(t), t, theta)

    Refer to: https://arxiv.org/pdf/1806.07366
    """

    def __init__(
        self,
        hidden_dims_f: list[int],
        hidden_dims_g: list[int] | None = None,
        dim_cov: int = 0,
    ):
        super().__init__()

        # Check if covariates are used
        self.dim_cov = dim_cov
        self.use_covariates = hidden_dims_g is not None

        # Dynamics network (f)
        # Approximates dz(t)/dt
        input_f = 4 + dim_cov if self.use_covariates else 4

        layers_f = []
        in_dim = input_f
        for h_dim in hidden_dims_f:
            layers_f.append(nn.Linear(in_dim, h_dim))
            layers_f.append(nn.Softplus())
            in_dim = h_dim

        layers_f.append(nn.Linear(in_dim, 1))
        self.net_f = nn.Sequential(*layers_f)

        # Covariate projection network (g)
        # See: https://www.geeksforgeeks.org/deep-learning/what-is-a-projection-layer-in-the-context-of-neural-networks/
        if self.use_covariates:
            layers_g = []
            in_dim_g = dim_cov
            for h_dim in hidden_dims_g:  # pyright: ignore
                layers_g.append(nn.Linear(in_dim_g, h_dim))
                layers_g.append(nn.Softplus())
                in_dim_g = h_dim

            layers_g.append(nn.Linear(in_dim_g, 1))
            self.net_g = nn.Sequential(*layers_g)
        else:
            # If covariates is not used then use constant parameter for volume
            self.v_param = nn.Parameter(torch.Tensor([1, 0]))

    def forward(self, t: float, z: Tensor) -> Tensor:
        """
        Compute dz/dt
        """
        t_tensor = torch.as_tensor([t], device=z.device)  # pyright: ignore

        # Feature concatenation for the dynamics function
        # z (state), t (time), z0 (start), u (dose), v (covariates), n (admin #)
        if self.use_covariates:
            x = torch.cat([z, t_tensor, self.z0, self.u, self.v, self.n_admin])
        else:
            x = torch.cat([z, t_tensor, self.z0, self.u, self.n_admin])

        return self.net_f(x)
