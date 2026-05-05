from data import PKData
from model import PKNODE
import torch
from torchdiffeq import odeint
import torch.nn.functional as F
import utils

# Loss Function
def loss_fn(pred, true):
    return F.mse_loss(pred, true)

# Train Funtion
def train(model, data, config, device):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["settings"]["train"]["learning_rate"]
    )

    epochs = config["settings"]["train"]["train_epoch"]

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for pid in data.patients:

            sample = data.get_patient_data(pid)
            times = sample["times"]
            y_true = sample["conc"]

            if len(times) < 2:
                continue
            
            times = torch.tensor(times, dtype=torch.floatt32, device=device)
            y_true = torch.tensor(y_true, dtype-torch.float32, device=device)

            # Initial condition (first concentration)
            z0 = y_true[0].unsqueeze(0)

            # ODE solve
            y_pred = odeint(
                lambda t, z: model(t, z),
                z0,
                times
            )

            y_pred = y_pred.squeeze()

            # Loss
            loss = loss_fn(y_pred, y_true)

            optimizer.zero(y_pred, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss: .4f}")


def main():
    config = utils.getConfig()

    # Model initialization, try to use CUDA/GPU if available
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

    train(model, data, config, device)


if __name__ == "__main__":
    main()
