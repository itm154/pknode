import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

matplotlib.use("Agg")  # prevent GUI pop-up

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load config file
def load_data(config_path):

    config = toml.load(config_path)

    # Read dataset path from config
    train_path = config["data"]["train_file"]
    test_path = config["data"]["test_file"]
    col_config = config["data"]["columns"]

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df, col_config


# Auto detect important columns
def clean_data(df, target_col):

    df = df.copy()

    # Clean data (Convert "." to NaN)
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


# Evaluation (Metrics)
def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# Plot 1: TRUE vs PREDICTED
def plot_results(name, y_true, y_pred):

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
    filename = f"ml_plots/models/{name.replace(' ', '_').lower()}_true_vs_pred.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    # Residual
    residuals = y_true - y_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Redisuals")
    filename = f"ml_plots/residuals/{name.replace(' ', '_').lower()}_residuals.png"
    plt.savefig(filename, dpi=300)
    plt.close()


# Plot 2: PK Curve
def plot_pk_curve_for_all(df, id_col, time_col, target_col):

    if id_col is None or time_col is None:
        return

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    for patient_id in df[id_col].dropna().unique():
        patient_df = df[df[id_col] == patient_id].copy()
        patient_df = patient_df.dropna(subset=[time_col, target_col])
        patient_df = patient_df.sort_values(time_col)

        plt.figure(figsize=(8, 5))
        plt.plot(patient_df[time_col], patient_df[target_col], marker="o")

        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.title(f"PK Curve - Patient {patient_id}")
        plt.grid(True)

        filename = f"ml_plots/pk_curves/pk_curve_{patient_id}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


# Main
def main():

    train_df, test_df, col_config = load_data("config.toml")
    target_col = col_config["conc"]
    # Clean both datasets
    train_df, target_col = clean_data(train_df, target_col)
    test_df, target_col = clean_data(test_df, target_col)

    print("Target Column: ", target_col)

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

        evaluate_model(name, y_test, y_pred)
        plot_results(name, y_test, y_pred)

    id_col = col_config.get("id", None)
    time_col = col_config.get("time", None)

    plot_pk_curve_for_all(train_df, id_col, time_col, target_col)


if __name__ == "__main__":
    main()
