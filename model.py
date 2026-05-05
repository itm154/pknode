# pyright: reportPrivateImportUsage=false
# pyright: reportArgumentType=false

import torch
import torch.nn as nn
from torch import Tensor


class PKNODE(nn.Module):
    """
    NODE model definition for PK/PD
    """

    def __init__(
        self,
        dim_c: list[int],
        dim_V: list[int] | None = None,
        dim_cov: int = 0,
    ):
        super().__init__()

        # Check if covariates are used
        self.include_covariates = dim_V is not None
        self.dim_cov = dim_cov

        # Dynamics network
        # Approximates dc(t)/dt
        input_c = 4 + dim_cov if self.include_covariates else 4
        layers_c = []
        in_dim = input_c
        for dim in dim_c:
            layers_c.append(nn.Linear(in_dim, dim))
            layers_c.append(nn.Softplus())
            in_dim = dim

        layers_c.append(nn.Linear(in_dim, 1))
        self.net_c = nn.Sequential(*layers_c)

        # Covariate projection network
        if self.include_covariates:
            layers_V = []
            in_dim_V = dim_cov
            for dim in dim_V:  # pyright: ignore
                layers_V.append(nn.Linear(in_dim_V, dim))
                layers_V.append(nn.Softplus())
                in_dim_V = dim

            layers_V.append(nn.Linear(in_dim_V, 1))
            self.net_V = nn.Sequential(*layers_V)
        else:
            # If covariates is not used then use constant parameter for volume
            self.V_param = nn.Parameter(torch.as_tensor([1.0], dtype=torch.float32))

    def forward(self, t: float, z: Tensor) -> Tensor:
        """
        Forward Pass
        """
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor([t], device=z.device, dtype=z.dtype)
        else:
            t_tensor = t.view(1)

        # Feature concatenation for the dynamics function
        if self.include_covariates:
            x = torch.cat([z, t_tensor, self.z0, self.n_admin, self.v])
        else:
            x = torch.cat([z, t_tensor, self.z0, self.n_admin])

        return self.net_c(x)
