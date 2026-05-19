import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # prevent GUI pop-up

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import getConfig # Reuse utils

# Create output directories for plots
def create_dirs():
    dirs = [
        "ml_plots/models",
        "ml_plots/residuals",
        "ml_plots/pk_curves"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# Clean data
def clean_data(df):

    df = df.copy()

    # Convert "." to NaN
    df.replace(".", np.nan, inplace=True)

    # Auto detect column
    possible_targets = ["DV", "CP", "CONC"]
    target_col = next((c for c in possible_targets if c in df.columns), None)

    if target_col is None:
        raise ValueError("No concentration column found.")

    print("Target Column: ", target_col)

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    return df, target_col


# Build features
def build_features(df, target_col):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols]

    return X, y, numeric_cols


# Preprocessor
def get_preprocessor(numeric_cols):

    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        ]
    )


# Extra Metrics (NODE-style)
def extra_metrics(y_true, y_pred):

    eps = 1e-8

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    pearson = np.corrcoef(y_true, y_pred)[0,  1]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "SMAPE": smape,
        "Pearson": pearson
    }

def print_metrics(name, metrics):
    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


# Plots
def plot_results(name, y_true, y_pred):

    # True vs Pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)

    # Diagonal reference line
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    # Prediction line
    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title(f"{name}: True vs Predicted")

    os.makedirs("ml_plots/models", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"ml_plots/models/{name}_true_vs_pred.png", dpi=300)
    plt.close()

    # Residual
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(name + "Redisuals")

    os.makedirs("ml_plots/residuals", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"ml_plots/residuals/{name}_residuals.png", dpi=300)
    plt.close()


# Plot 2: PK Curves (All patients)
def plot_pk_curve_for_all(df, id_col, time_col, target_col):

    if id_col is None or time_col is None:
        return

    os.makedirs("ml_plots/pk_curves", exist_ok=True)

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    for pid in df[id_col].dropna().unique():

        patient = df[df[id_col] == pid].copy()
        patient = patient.dropna(subset=[time_col, target_col])
        patient = patient.sort_values(time_col)

        if len(patient) < 2:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(patient[time_col], patient[target_col], marker="o")

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title(f"PK Curve - Patient {pid}")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"ml_plots/pk_curves/pk_{pid}.png", dpi=300)
        plt.close()


# Main
def main(config_path):

    create_dirs()

    # Load config via utils
    config = getConfig(config_path)

    train_path = config["data"]["train_file"]
    test_path = config["data"]["test_file"]
    col_config = config["data"]["columns"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean both datasets
    train_df, target_col = clean_data(train_df)
    test_df, _ = clean_data(test_df)

    # Build train
    X_train, y_train, numeric_cols = build_features(train_df, target_col)

    # Build test using SAME columns
    X_test = test_df[numeric_cols]
    y_test = test_df[target_col]

    preprocessor = get_preprocessor(numeric_cols)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42, n_jobs=-1
        ),
    }

    for name, model in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = extra_metrics(y_test, y_pred)
        print_metrics(name, metrics)

        plot_results(name, y_test, y_pred)

    id_col = col_config.get("id", None)
    time_col = col_config.get("time", None)

    plot_pk_curve_for_all(train_df, id_col, time_col, target_col)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.toml")

    args = parser.parse_args()

    main(args.config)
