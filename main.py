from data import PKData
from model import PKNODE
import torch
import utils


def main():
    config = utils.getConfig()

    # Model initialization, try to use CUDA/GPU if available
    # Idk if it would work on Windows without additional steps, only tested on Linux
    dim_c = config["settings"]["nn"]["dim_c"]
    dim_V = config["settings"]["nn"]["dim_V"]
    n_cov = len(config["data"]["columns"]["covariates"])
    model = PKNODE(dim_c, dim_V, n_cov)

    device = (
        torch.accelerator.current_accelerator().type  # pyright: ignore
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.to(device)
    print(f"Using {device} device")
    print(model)


if __name__ == "__main__":
    main()
