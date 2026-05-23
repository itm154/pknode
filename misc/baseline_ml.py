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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import getConfig  # Reuse utils


# Create output directories for plots
def create_dirs():
    base_dir = "misc/ml_plots"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(os.path.join(base_dir, "pk_curves"), exist_ok=True)


# Clean and Engineer Data
def clean_and_engineer_data(df, col_config):
    df = df.copy()
    df.replace(".", np.nan, inplace=True)

    target_col = col_config.get("conc")
    if target_col not in df.columns:
        possible_targets = ["DV", "CP", "CONC"]
        target_col = next((c for c in possible_targets if c in df.columns), None)

    if target_col is None:
        raise ValueError("No concentration column found.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    id_col = col_config.get("id", "ID")
    time_col = col_config.get("time", "TIME")
    dose_col = col_config.get("dose", "DOSE")

    # Basic cleaning
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
    if dose_col in df.columns:
        df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0)

    # Feature Engineering
    # 1. Time Since Last Dose (TSLD)
    df = df.sort_values([id_col, time_col])

    is_dose = (df["EVID"] == 1) if "EVID" in df.columns else (df[dose_col] > 0)
    df["DOSE_TIME"] = np.where(is_dose, df[time_col], np.nan)
    df["LAST_DOSE_TIME"] = df.groupby(id_col)["DOSE_TIME"].ffill()
    df["TSLD"] = df[time_col] - df["LAST_DOSE_TIME"]
    df["TSLD"] = df["TSLD"].fillna(df[time_col])  # Fallback to time if no dose yet

    # 2. Cumulative Dose
    df["CUM_DOSE"] = df.groupby(id_col)[dose_col].cumsum()

    # 3. Last Dose Amount
    df["LAST_DOSE_AMT"] = np.where(is_dose, df[dose_col], np.nan)
    df["LAST_DOSE_AMT"] = df.groupby(id_col)["LAST_DOSE_AMT"].ffill().fillna(0)

    # Filter for observations (EVID=0) for training/testing
    evid_col = col_config.get("evid")
    if evid_col and evid_col in df.columns:
        df_clean = df[df[evid_col] == 0].copy()
    else:
        df_clean = df[df[target_col].notna() & (df[dose_col] == 0)].copy()

    return df_clean, target_col, df  # Return full df for PK curve plotting


# Build features
def build_features(df, target_col):
    X = df.drop(columns=[target_col])

    # Identify relevant features
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Remove identifiers that shouldn't be features
    for col in ["ID", "EVID", "DOSE_TIME", "LAST_DOSE_TIME"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    X = X[numeric_cols]
    return X, df[target_col], numeric_cols


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


def plot_results(y_true, y_pred):
    model_dir = "misc/ml_plots"

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)

    all_min = np.min([plt.xlim()[0], plt.ylim()[0]])
    all_max = np.max([plt.xlim()[1], plt.ylim()[1]])
    plt.plot([all_min, all_max], [all_min, all_max], "r--", alpha=0.75, zorder=0)

    plt.xlabel("Observed Concentration")
    plt.ylabel("Predicted Concentration")
    plt.title("Predicted vs Observed")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(model_dir, "pred_vs_obs.png"), dpi=300)
    plt.close()


def plot_model_pk_curves(pipeline, df_full, id_col, time_col, dose_col, target_col):
    if id_col is None or time_col is None:
        return

    pk_dir = "misc/ml_plots/pk_curves"

    # Predict for the whole set (including administrations) to see the curve shape
    # We use the full engineered dataframe
    X_plot, _, _ = build_features(df_full, target_col)
    df_plot = df_full.copy()
    df_plot["PRED"] = pipeline.predict(X_plot)

    for pid in df_plot[id_col].dropna().unique():
        patient = df_plot[df_plot[id_col] == pid].sort_values(time_col)
        if len(patient) < 1:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(
            patient[time_col], patient["PRED"], label="ML Prediction", color="blue"
        )

        # Add vertical lines for administrations
        dose_times = patient[patient[dose_col] > 0][time_col]
        for dt in dose_times:
            plt.axvline(
                x=dt,
                color="gray",
                linestyle="--",
                alpha=0.4,
                label="Administration" if dt == dose_times.iloc[0] else "",
            )

        # Only scatter points that were actually observations
        is_obs = (
            (patient["EVID"] == 0)
            if "EVID" in patient.columns
            else (patient[target_col] > 0)
        )
        obs = patient[is_obs]

        if not obs.empty:
            plt.scatter(
                obs[time_col],
                obs[target_col],
                color="red",
                label="Ground Truth",
                zorder=5,
            )

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title(f"Patient Drug Concentration Prediction for ID: {pid}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(pk_dir, f"{pid}.png"), dpi=300)
        plt.close()


def main(config_path):
    config = getConfig(config_path)
    train_path, test_path = config["data"]["train_file"], config["data"]["test_file"]
    col_config = config["data"]["columns"]

    if not os.path.exists(train_path):
        train_path = os.path.join("..", train_path)
    if not os.path.exists(test_path):
        test_path = os.path.join("..", test_path)

    # Load and engineer
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    train_df, target_col, _ = clean_and_engineer_data(train_raw, col_config)
    test_df, _, test_df_full = clean_and_engineer_data(test_raw, col_config)

    X_train, y_train, numeric_cols = build_features(train_df, target_col)
    X_test, y_test, _ = build_features(test_df, target_col)

    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=10, learning_rate=0.05, random_state=42
    )

    create_dirs()
    id_col, time_col, dose_col = (
        col_config.get("id"),
        col_config.get("time"),
        col_config.get("dose"),
    )

    pipeline = Pipeline(
        [("preprocessor", get_preprocessor(numeric_cols)), ("model", model)]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = calculate_all_metrics(y_test, y_pred)
    print_metrics("Gradient Boosting", metrics)
    plot_results(y_test, y_pred)
    plot_model_pk_curves(pipeline, test_df_full, id_col, time_col, dose_col, target_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.toml")
    main(parser.parse_args().config)
