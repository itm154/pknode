# PKNODE

PKNODE is an implementation of Neural ODE (NODE) to predict drug concentration.
It is an experimental alternative to traditional models such as Non-Linear Mixed
Effects (NLME)

_NOT INTENDED FOR REAL WORLD USE_

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

### Model Training

#### Command-Line Arguments

| Argument       | Description                                                       |
| :------------- | :---------------------------------------------------------------- |
| `-c, --config` | Path to the TOML configuration file (default: `config.toml`).     |
| `--resume`     | Resume training from the most recent checkpoint.                  |
| `--load-model` | Load a specific `.pth` model file for fine-tuning or re-training. |

#### Examples

**Standard training execution:**

```bash
uv run train.py -c examples/theophylline.toml
```

**Resume training from checkpoint:**

```bash
uv run train.py -c examples/theophylline.toml --resume
```

**Load an existing model to further train it:**

```bash
uv run train.py -c examples/theophylline.toml --load-model models/theophylline.pth
```

---

### Model Evaluation

#### Command-Line Arguments

| Argument       | Description                                                                                                     |
| :------------- | :-------------------------------------------------------------------------------------------------------------- |
| `-c, --config` | Path to the TOML configuration file.                                                                            |
| `--save-plots` | Generate and save concentration-time profiles (including 90% prediction intervals) and residual analysis plots. |

#### Examples

**Basic evaluation:**

```bash
uv run eval.py -c examples/theophylline.toml
```

**Full evaluation with visualization:**

```bash
uv run eval.py -c examples/theophylline.toml --save-plots
```

<details>
<summary><b>Configuration Details</b></summary>

The project uses TOML files for configuration.

#### `[model]`

- `name` (string): Name of the model (used for filenames).
- `path` (string): Directory to save models.
- `absorption` (boolean): `true` for first-order absorption (ADME), `false` for
  instantaneous absorption (DME).

#### `[data]`

- `train_file` (string): Path to training CSV.
- `test_file` (string): Path to testing CSV.

#### `[data.columns]`

- `id` (string): Patient ID column.
- `time` (string): Time column.
- `dose` (string): Dose amount column.
- `conc` (string): Concentration column.
- `evid` (string, optional): Event ID column.
- `covariates` (list of strings): List of covariate column names.

#### `[settings.train]`

- `epoch` (integer): Training epochs.
- `learning_rate` (float): Initial learning rate.
- `weight_decay` (float): Regularization strength.
- `patience` (integer): Epochs to wait before LR reduction.
- `factor` (float): LR reduction factor.

#### `[settings.nn]`

- `dim_c` (list of integers): Layers for dynamics network (e.g.,
  `[32, 32, 32]`).
- `dim_V` (list of integers): Layers for Volume network (e.g., `[16]`).
- `include_covariates` (boolean): Enable/disable covariate usage.

#### Example: `theophylline.toml`

```toml
[model]
name = "theophylline"
path = "./models"
absorption = true

[data]
train_file = "./data/theophylline_train.csv"
test_file = "./data/theophylline_test.csv"

[data.columns]
id = "ID"
time = "TIME"
dose = "AMT"
conc = "CONC"
covariates = ["WEIGHT", "SEX"]

[settings.train]
epoch = 100
weight_decay = 1e-4
learning_rate = 1e-3
patience = 5
factor = 0.5

[settings.nn]
dim_c = [64, 64, 64]
dim_V = [16]
include_covariates = true
```

</details>

## Credits

- This project is based on
  [TommyGiak/pharmacoNODE](https://github.com/TommyGiak/pharmacoNODE). This
  project's purpose is to improve and build upon the original.
- Datasets obtained from:
  - [nlmixr2data](https://nlmixr2.github.io/nlmixr2data/)
  - [Monolix](https://monolixsuite.slp-software.com/monolix/2024R1/data-set-examples)

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for
details.
