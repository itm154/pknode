# pyright: reportPrivateImportUsage=false
# pyright: reportArgumentType=false

import torch
import torch.nn as nn
from torch import Tensor


# Read: https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
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

        # Define the layers of the neural network

        # Dynamics network
        # Approximates dc(t)/dt
        input_c = 4 + dim_cov if self.include_covariates else 4
        layers_c = []
        in_dim = input_c
        for dim in dim_c:
            layers_c.append(nn.Linear(in_dim, dim))
            layers_c.append(nn.SiLU())
            in_dim = dim

        layers_c.append(nn.Linear(in_dim, 1))
        self.net_c = nn.Sequential(*layers_c)

        # Covariate projection network
        if self.include_covariates:
            layers_V = []
            in_dim_V = dim_cov
            if dim_V:
                for dim in dim_V:
                    layers_V.append(nn.Linear(in_dim_V, dim))
                    layers_V.append(nn.SiLU())
                    in_dim_V = dim

            layers_V.append(nn.Linear(in_dim_V, 1))
            layers_V.append(nn.Softplus())  # Ensure Volume is always positive
            self.net_V = nn.Sequential(*layers_V)
        else:
            # If covariates is not used then use constant parameter for volume
            self.V_param_internal = nn.Parameter(
                torch.as_tensor([1.0], dtype=torch.float32)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.net_c[-1].weight)
        nn.init.zeros_(self.net_c[-1].bias)

        if self.include_covariates:
            nn.init.zeros_(self.net_V[-2].weight)
            nn.init.zeros_(self.net_V[-2].bias)

    @property
    def V_param(self):
        if hasattr(self, "V_param_internal"):
            return torch.nn.functional.softplus(self.V_param_internal)
        return None

    def forward(self, t: float, z: Tensor) -> Tensor:
        """
        Forward Pass
        """

        # Just to supress torch's warning
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor([t], device=z.device, dtype=z.dtype)
        else:
            t_tensor = t.view(1)

        # Ensure concentration doesn't go negative
        z_safe = torch.clamp(z, min=0.0)

        # Feature concatenation for the dynamics function
        if self.include_covariates:
            x = torch.cat([z_safe, t_tensor, self.z0, self.n_admin, self.v])
        else:
            x = torch.cat([z_safe, t_tensor, self.z0, self.n_admin])

        return self.net_c(x)
