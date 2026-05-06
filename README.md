# PKNODE: Pharmacokinetic Neural Ordinary Differential Equations

PKNODE uses Neural Ordinary Differential Equations (Neural ODEs) for drug
concentration prediction. It is an alternative to traditional compartmental
models and Non-Linear Mixed Effects (NLME) models.

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/): Preferred Python package and project
  manager.
- (Optional but Highly recommended) NVIDIA GPU for accelerated training.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/itm154/pknode.git
   cd pknode
   ```
   or use GitHub Desktop

2. **Initialize the environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```
   If you're not using uv:
   ```bash
   python -m venv .venv
   # Follow the instructions to activate the venv
   python -m pip install -r requirements.txt
   ```

## Usage

### 1. Configure the Model

Edit `config.toml` to define your dataset paths, column mappings, and model
parameters. (Project is still under development, configuration is bound to
change)

<details>
<summary>View Example config.toml</summary>

```toml
[model]
name = "tobramycin"
path = "./models"

[data]
file = "./data/tobramycin.csv"
inc_cov = true

[data.columns]
id = "ID"
time = "TIME"
dose = "DOSE"
conc = "CP"
evid = "EVID"
covariates = ["AGE", "SEX", "CLCR"]

[settings.train]
pre_train_epoch = 10
train_epoch = 10
finetune_epoch = 10
weight_decay = 0
learning_rate = 1e-3

[settings.nn]
dim_c = [20, 20, 20]
dim_V = [10]
include_covariates = true
```

</details>

<details>
<summary>Detailed Configuration Explanation</summary>

#### [model]

- **name**: String. The filename used when saving the trained model (e.g.,
  "warfarin" saves as `warfarin.pth`).
- **path**: String. The directory where model weights will be saved.

#### [data]

- **file**: String. Path to the CSV dataset.
- **inc_cov**: Boolean. Global flag indicating if the dataset includes covariate
  columns.

#### [data.columns]

- **id**: String. Column name for unique patient/subject identifiers.
- **time**: String. Column name for the time of the event (administration or
  observation).
- **dose**: String. Column name for the amount of drug administered (e.g., AMT).
- **conc**: String. Column name for the measured drug concentration (e.g., DV or
  CP).
- **evid**: String (Optional). Column name for the event ID (1 for dose, 0 for
  observation). If omitted, the code will infer event types from the dose and
  concentration columns.
- **covariates**: List of Strings. Names of columns containing patient-specific
  features like weight, age, or sex.

#### [settings.train]

- **train_epoch**: Integer. Number of times the model iterates through the full
  dataset during training.
- **learning_rate**: Float. The step size used by the Adam optimizer.
- **weight_decay**: Float. L2 regularization factor used to prevent overfitting.
- **pre_train_epoch**: Integer. Number of epochs for initial synthetic or
  warm-up training.
- **finetune_epoch**: Integer. Number of epochs for final model refinement.

#### [settings.nn]

- **dim_c**: List of Integers. Defines the hidden layers for the dynamics
  network (approximating $dc/dt$). For example, `[20, 20]` creates two layers
  with 20 neurons each.
- **dim_V**: List of Integers. Defines the hidden layers for the covariate
  projection network that predicts the Volume of Distribution ($V$).
- **include_covariates**: Boolean. Set to `true` to enable the model to use
  patient features to adjust predictions.

</details>

### 2. Run Training

```bash
uv run train.py
# or
python train.py
```

## Credits

- This project is based on
  [TommyGiak/pharmacoNODE](https://github.com/TommyGiak/pharmacoNODE). This
  project's purpose is to improve and build upon the original.

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for
details.
