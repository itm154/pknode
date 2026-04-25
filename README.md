# PKNODE: Pharmacokinetic Neural Ordinary Differential Equations

PKNODE uses Neural Ordinary Differential Equations (Neural ODEs) for drug
concentration prediction. It is an alternative to traditional compartmental
models and Non-Linear Mixed Effects (NLME) models.

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/): Preferred Python package and project
  manager.
- (Optional) NVIDIA GPU for accelerated training.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/itm154/pknode.git
   cd pknode
   ```

2. **Initialize the environment and install dependencies:**
   ```bash
   uv venv
   uv sync
   ```

## Usage

### 1. Configure the Model

Edit `config.toml` to define your dataset paths, column mappings, and model
parameters.

```toml
[data]
file = "./data/your_dataset.csv"

[data.columns]
id = "ID"
time = "TIME"
dose = "DOSE"
conc = "CP"
evid = "EVID"
covariates = ["AGE", "SEX", "CLCR"]
```

### 2. Run Training/Prediction (TODO)

```bash
uv run main.py
```

## Credits

- This project is based on
  [TommyGiak/pharmacoNODE](https://github.com/TommyGiak/pharmacoNODE). This
  project's purpose is to improve and build upon the original.

## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for
details.
