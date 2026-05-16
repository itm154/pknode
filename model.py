import torch
import torch.nn as nn
from torch import Tensor


# Read: https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
class PKNODE(nn.Module):
    """
    NODE model definition for PK/PD
    """

    z0: Tensor
    v: Tensor
    time_scale: Tensor
    z0_scale: Tensor
    cov_means: Tensor
    cov_stds: Tensor

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

        # Dynamic scales for input normalization
        self.register_buffer("time_scale", torch.tensor(1.0))
        self.register_buffer("z0_scale", torch.tensor(1.0))

        if self.include_covariates:
            self.register_buffer("cov_means", torch.zeros(dim_cov))
            self.register_buffer("cov_stds", torch.ones(dim_cov))

        # Dynamics network
        # Approximates dc(t)/dt
        input_c = 3 + dim_cov if self.include_covariates else 3
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
                nn.init.zeros_(m.bias)

        last_layer_c = self.net_c[-1]
        if isinstance(last_layer_c, nn.Linear):
            nn.init.zeros_(last_layer_c.weight)
            nn.init.zeros_(last_layer_c.bias)

        if self.include_covariates:
            penultimate_layer_V = self.net_V[-2]
            if isinstance(penultimate_layer_V, nn.Linear):
                nn.init.zeros_(penultimate_layer_V.weight)
                nn.init.zeros_(penultimate_layer_V.bias)

    @property
    def V_param(self):
        if hasattr(self, "V_param_internal"):
            return torch.nn.functional.softplus(self.V_param_internal)
        return None

    def predict_V(self, v: Tensor | None = None) -> Tensor:
        """
        Predict Volume (V)
        """
        if self.include_covariates and v is not None:
            v_scaled = (v - self.cov_means) / (self.cov_stds + 1e-7)
            return self.net_V(v_scaled)
        return self.V_param

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

        # Feature concatenation for the dynamics function with scaling
        if self.include_covariates:
            v_scaled = (self.v - self.cov_means) / (self.cov_stds + 1e-7)
            x = torch.cat(
                [
                    z_safe,
                    t_tensor / self.time_scale,
                    self.z0 / self.z0_scale,
                    v_scaled,
                ]
            )
        else:
            x = torch.cat(
                [
                    z_safe,
                    t_tensor / self.time_scale,
                    self.z0 / self.z0_scale,
                ]
            )

        return -torch.nn.functional.softplus(self.net_c(x)) * z_safe
