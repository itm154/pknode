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
train_file = "./data/tobramycin_train.csv"
test_file = "./data/tobramycin_test.csv"

[data.columns]
id = "ID"
time = "TIME"
dose = "DOSE"
conc = "CP"
evid = "EVID"
covariates = ["AGE", "SEX", "CLCR"]

[settings.train]
epoch = 30
weight_decay = 1e-4
learning_rate = 5e-4
step_size = 10
gamma = 0.5

[settings.nn]
dim_c = [32, 32, 32]
dim_V = [16]
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

- **train_file**: String. Path to the CSV training dataset.
- **test_file**: String. Path to the CSV testing dataset.

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

- **epoch**: Integer. Number of times the model iterates through the full
  dataset during training.
- **weight_decay**: Float. L2 regularization factor used to prevent overfitting.
- **learning_rate**: Float. The step size used by the AdamW optimizer.
- **step_size**: Integer. Step size used by the scheduler.
- **gamma**: Float. Factor to multiply by with learning rate with every
  step_size.

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

Specify configuration to use (default is `config.toml`):

```bash
uv run train.py --config examples/warfarin.toml
```

## Credits

- This project is based on
  [TommyGiak/pharmacoNODE](https://github.com/TommyGiak/pharmacoNODE). This
  project's purpose is to improve and build upon the original.

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for
details.
