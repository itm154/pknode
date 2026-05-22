import argparse
import os
import shutil
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # prevent GUI pop-up

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import getConfig  # Reuse utils


# Create output directories for plots
def create_dirs(model_names):
    base_dir = "misc/ml_plots"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    for name in model_names:
        folder_name = name.lower().replace(" ", "_")
        os.makedirs(os.path.join(base_dir, folder_name, "pk_curves"), exist_ok=True)


# Clean data
def clean_data(df, col_config):
    df = df.copy()
    df.replace(".", np.nan, inplace=True)

    target_col = col_config.get("conc")
    if target_col not in df.columns:
        possible_targets = ["DV", "CP", "CONC"]
        target_col = next((c for c in possible_targets if c in df.columns), None)

    if target_col is None:
        raise ValueError("No concentration column found.")

    print("Target Column: ", target_col)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Filter for observations (EVID=0) if column exists
    evid_col = col_config.get("evid")
    if evid_col and evid_col in df.columns:
        df = df[df[evid_col] == 0]
    else:
        # Fallback: remove rows where concentration is NaN
        df = df.dropna(subset=[target_col])
        # If no EVID, rows with dose > 0 are administrations
        dose_col = col_config.get("dose")
        if dose_col and dose_col in df.columns:
            df = df[pd.to_numeric(df[dose_col], errors="coerce").fillna(0) == 0]

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


def calculate_all_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Linear scale metrics
    mse_lin = mean_squared_error(y_true, y_pred)
    rmse_lin = np.sqrt(mse_lin)
    mae_lin = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    r2_lin = r2_score(y_true, y_pred)

    # Log scale metrics (MSLE)
    log_true = np.log(np.maximum(y_true, 1e-7))
    log_pred = np.log(np.maximum(y_pred, 1e-7))
    msle = np.mean((log_true - log_pred) ** 2)

    # R-squared (Log)
    ss_res_log = np.sum((log_true - log_pred) ** 2)
    ss_tot_log = np.sum((log_true - np.mean(log_true)) ** 2)
    r2_log = 1 - (ss_res_log / (ss_tot_log + 1e-7))

    return {
        "MAE (Linear)": mae_lin,
        "MAPE (%)": mape,
        "RMSE (Linear)": rmse_lin,
        "MSLE (Log)": msle,
        "R-squared (Linear)": r2_lin,
        "R-squared (Log)": r2_log,
    }


def print_metrics(name, metrics):
    print(f"\nEvaluation Results for {name}:")
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    print(f"{'RMSE (Linear)':<20} | {metrics['RMSE (Linear)']:.4e}")
    print(f"{'MAE (Linear)':<20} | {metrics['MAE (Linear)']:.4e}")
    print(f"{'MAPE (%)':<20} | {metrics['MAPE (%)']:.2f}%")
    print(f"{'R-squared (Linear)':<20} | {metrics['R-squared (Linear)']:.4f}")
    print(f"{'MSLE (Log)':<20} | {metrics['MSLE (Log)']:.4e}")
    print(f"{'R-squared (Log)':<20} | {metrics['R-squared (Log)']:.4f}")


def plot_results(name, y_true, y_pred):
    folder_name = name.lower().replace(" ", "_")
    model_dir = os.path.join("misc/ml_plots", folder_name)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title(f"{name}: True vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "true_vs_pred.png"), dpi=300)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name} Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "residuals.png"), dpi=300)
    plt.close()


def plot_model_pk_curves(name, pipeline, df, id_col, time_col, target_col):
    if id_col is None or time_col is None:
        return

    folder_name = name.lower().replace(" ", "_")
    pk_dir = os.path.join("misc/ml_plots", folder_name, "pk_curves")

    df_plot = df.copy()
    df_plot[time_col] = pd.to_numeric(df_plot[time_col], errors="coerce")

    # Predict for the whole set
    X, _, _ = build_features(df_plot, target_col)
    df_plot["PRED"] = pipeline.predict(X)

    for pid in df_plot[id_col].dropna().unique():
        patient = df_plot[df_plot[id_col] == pid].sort_values(time_col)
        if len(patient) < 1:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(
            patient[time_col], patient["PRED"], label=f"{name} Prediction", color="blue"
        )
        plt.scatter(
            patient[time_col],
            patient[target_col],
            color="red",
            label="Ground Truth",
            zorder=5,
        )

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title(f"{name} PK Curve - Patient {pid}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pk_dir, f"pk_{pid}.png"), dpi=300)
        plt.close()


def main(config_path):
    config = getConfig(config_path)
    train_path, test_path = config["data"]["train_file"], config["data"]["test_file"]
    col_config = config["data"]["columns"]

    if not os.path.exists(train_path):
        train_path = os.path.join("..", train_path)
    if not os.path.exists(test_path):
        test_path = os.path.join("..", test_path)

    train_df, target_col = clean_data(pd.read_csv(train_path), col_config)
    test_df, _ = clean_data(pd.read_csv(test_path), col_config)

    X_train, y_train, numeric_cols = build_features(train_df, target_col)
    X_test, y_test = test_df[numeric_cols], test_df[target_col]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    create_dirs(models.keys())
    id_col, time_col = col_config.get("id"), col_config.get("time")

    for name, model in models.items():
        pipeline = Pipeline(
            [("preprocessor", get_preprocessor(numeric_cols)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        metrics = calculate_all_metrics(y_test, y_pred)
        print_metrics(name, metrics)
        plot_results(name, y_test, y_pred)
        plot_model_pk_curves(name, pipeline, test_df, id_col, time_col, target_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.toml")
    main(parser.parse_args().config)
