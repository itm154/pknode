import argparse
import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from utils import getConfig


def run_individual_pipeline(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # =====================================================================
    # 1. PK DATA CLEANING & PROPAGATION
    # =====================================================================
    df["ID"] = df["ID"].astype(str)
    df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce").fillna(0.0)
    df["DOSE_NUM"] = pd.to_numeric(df["DOSE"], errors="coerce").fillna(0.0)

    target_col = "CP" if "CP" in df.columns else "DV"
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)

    is_dose_row = (df["EVID"] == 1) if "EVID" in df.columns else (df["DOSE_NUM"] > 0)

    df["VALID_DOSE"] = np.where(is_dose_row, df["DOSE_NUM"], np.nan)
    df["DOSE_TIME"] = np.where(is_dose_row, df["TIME"], np.nan)

    df["LAST_DOSE"] = df.groupby("ID")["VALID_DOSE"].ffill().fillna(0.0)
    df["LAST_DOSE_TIME"] = df.groupby("ID")["DOSE_TIME"].ffill().fillna(0.0)
    df["TSLD"] = df["TIME"] - df["LAST_DOSE_TIME"]

    df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce").fillna(df["AGE"].median())
    df["AGE_SCALED"] = df["AGE"].values / 60.0

    if "CLCR" in df.columns:
        df["CLCR"] = pd.to_numeric(df["CLCR"], errors="coerce").fillna(
            df["CLCR"].median()
        )
        cov_scaled = df["CLCR"].values / 100.0
        cov_type = "CLCR"
    elif "WT" in df.columns:
        df["WT"] = pd.to_numeric(df["WT"], errors="coerce").fillna(df["WT"].median())
        cov_scaled = df["WT"].values / 70.0
        cov_type = "WT"
    else:
        cov_scaled = np.ones(len(df))
        cov_type = "None"

    # =====================================================================
    # 2. BUILD THE PYMC MODEL
    # =====================================================================
    patient_idx, patients = pd.factorize(df["ID"])
    n_patients = len(patients)
    obs_mask = (df["EVID"] == 0) if "EVID" in df.columns else (df[target_col] > 0)
    obs_mask = obs_mask.values

    with pm.Model() as pop_pk_model:
        cl_pop = pm.LogNormal("cl_pop", mu=np.log(4.0), sigma=0.3)
        v_pop = pm.LogNormal("v_pop", mu=np.log(20.0), sigma=0.3)
        ka_pop = pm.LogNormal("ka_pop", mu=np.log(1.2), sigma=0.3)
        beta_cov = pm.Normal(f"beta_{cov_type.lower()}", mu=1.0, sigma=0.5)
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=0.5)
        omega_cl = pm.HalfNormal("omega_cl", sigma=0.2)
        omega_v = pm.HalfNormal("omega_v", sigma=0.2)
        omega_ka = pm.HalfNormal("omega_ka", sigma=0.2)
        eta_cl_raw = pm.Normal("eta_cl_raw", mu=0, sigma=1, shape=n_patients)
        eta_v_raw = pm.Normal("eta_v_raw", mu=0, sigma=1, shape=n_patients)
        eta_ka_raw = pm.Normal("eta_ka_raw", mu=0, sigma=1, shape=n_patients)

        cl_i = pm.math.clip(
            cl_pop
            * (cov_scaled**beta_cov)
            * (df["AGE_SCALED"].values ** beta_age)
            * pm.math.exp(eta_cl_raw[patient_idx] * omega_cl),
            0.01,
            100.0,
        )
        v_i = pm.math.clip(
            v_pop * pm.math.exp(eta_v_raw[patient_idx] * omega_v), 0.5, 500.0
        )
        ka_i = pm.math.clip(
            ka_pop * pm.math.exp(eta_ka_raw[patient_idx] * omega_ka), 0.01, 10.0
        )

        k_elim = cl_i / v_i
        denom = pm.math.switch(pm.math.eq(ka_i - k_elim, 0), 1e-5, ka_i - k_elim)
        expected_cp = (df["LAST_DOSE"].values * ka_i / (v_i * denom)) * (
            pm.math.exp(pm.math.clip(-k_elim * df["TSLD"].values, -50, 0))
            - pm.math.exp(pm.math.clip(-ka_i * df["TSLD"].values, -50, 0))
        )

        sigma_proportional = pm.HalfNormal("sigma_proportional", sigma=0.2)
        sigma_additive = pm.HalfNormal("sigma_additive", sigma=0.1)
        sigma_total = pm.math.sqrt(
            sigma_additive**2 + (sigma_proportional * expected_cp[obs_mask]) ** 2
        )

        pm.Normal(
            "p_conc",
            mu=expected_cp[obs_mask],
            sigma=sigma_total,
            observed=df[target_col].values[obs_mask],
        )

        print("Finding MAP estimate...")
        map_estimate = pm.find_MAP(method="L-BFGS-B")

    return df, map_estimate, patients, cov_type


def plot_individual_profiles(df, map_estimate, patients, cov_type):
    print(
        f"Generating individual plots for {len(patients)} patients in misc/nlme_plots directory..."
    )
    os.makedirs("misc/nlme_plots", exist_ok=True)

    all_obs_list = []
    all_pred_list = []

    for pid in patients:
        plt.figure(figsize=(10, 6))

        patient_data = df[df["ID"] == pid].copy()
        obs_data = patient_data[patient_data["EVID"] == 0]
        dose_data = patient_data[patient_data["EVID"] == 1]

        # Ground truth points
        if not obs_data.empty:
            plt.scatter(
                obs_data["TIME"],
                obs_data["CP"],
                color="red",
                label="Observed",
                zorder=5,
            )

        # Dose markers
        for dose_time in dose_data["TIME"]:
            plt.axvline(x=dose_time, color="gray", linestyle="--", alpha=0.5)

        # Predicted curve
        t_min, t_max = patient_data["TIME"].min(), patient_data["TIME"].max()
        t_grid = np.linspace(t_min, t_max, 300)

        # Re-calculate parameters for this individual from MAP
        p_idx = np.where(patients == pid)[0][0]

        # Constants for the patient
        age_scaled = patient_data["AGE_SCALED"].iloc[0]
        cov_val = (
            (patient_data["CLCR"].iloc[0] / 100.0)
            if cov_type == "CLCR"
            else (patient_data["WT"].iloc[0] / 70.0 if cov_type == "WT" else 1.0)
        )

        cl_i = np.clip(
            map_estimate["cl_pop"]
            * (cov_val ** map_estimate[f"beta_{cov_type.lower()}"])
            * (age_scaled ** map_estimate["beta_age"])
            * np.exp(map_estimate["eta_cl_raw"][p_idx] * map_estimate["omega_cl"]),
            0.01,
            100.0,
        )
        v_i = np.clip(
            map_estimate["v_pop"]
            * np.exp(map_estimate["eta_v_raw"][p_idx] * map_estimate["omega_v"]),
            0.5,
            500.0,
        )
        ka_i = np.clip(
            map_estimate["ka_pop"]
            * np.exp(map_estimate["eta_ka_raw"][p_idx] * map_estimate["omega_ka"]),
            0.01,
            10.0,
        )

        k_elim = cl_i / v_i

        # Calculate predicted concentration on the grid (superposition of doses)
        cp_pred = np.zeros_like(t_grid)
        for _, dose in dose_data.iterrows():
            t_dose = dose["TIME"]
            amt = dose["DOSE"]

            tsld_grid = t_grid - t_dose
            valid_mask = tsld_grid >= 0

            denom = ka_i - k_elim
            if abs(denom) < 1e-5:
                denom = 1e-5

            # 1-comp oral equation contribution from this dose
            contribution = (amt * ka_i / (v_i * denom)) * (
                np.exp(-k_elim * tsld_grid[valid_mask])
                - np.exp(-ka_i * tsld_grid[valid_mask])
            )
            cp_pred[valid_mask] += contribution

        # Calculate predicted concentration at observation times for obs vs pred plot
        if not obs_data.empty:
            obs_times = obs_data["TIME"].values
            cp_obs_pred = np.zeros_like(obs_times)
            for _, dose in dose_data.iterrows():
                t_dose = dose["TIME"]
                amt = dose["DOSE"]
                tsld_obs = obs_times - t_dose
                valid_obs_mask = tsld_obs >= 0

                denom = ka_i - k_elim
                if abs(denom) < 1e-5:
                    denom = 1e-5

                contribution = (amt * ka_i / (v_i * denom)) * (
                    np.exp(-k_elim * tsld_obs[valid_obs_mask])
                    - np.exp(-ka_i * tsld_obs[valid_obs_mask])
                )
                cp_obs_pred[valid_obs_mask] += contribution

            all_obs_list.extend(obs_data["CP"].values)
            all_pred_list.extend(cp_obs_pred)

        plt.plot(t_grid, cp_pred, label="Predicted Curve", color="blue", linewidth=2)
        plt.title(f"Tobramycin Individual Profile - Patient ID: {pid}")
        plt.xlabel("Time (h)")
        plt.ylabel("Concentration (mg/L)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(f"misc/nlme_plots/patient_{pid}.png")
        plt.close()

    if all_obs_list:
        all_obs = np.array(all_obs_list)
        all_pred = np.array(all_pred_list)
        max_val = max(np.max(all_obs), np.max(all_pred)) * 1.05

        # Calculate and Print Metrics
        target_arr = all_obs
        preds_arr = all_pred

        log_preds = np.log(np.maximum(preds_arr, 1e-7))
        log_target = np.log(np.maximum(target_arr, 1e-7))
        log_residuals = log_target - log_preds

        # Linear scale metrics
        mse_lin = np.mean((preds_arr - target_arr) ** 2)
        rmse_lin = np.sqrt(mse_lin)
        mae_lin = np.mean(np.abs(preds_arr - target_arr))
        mape = np.mean(np.abs((target_arr - preds_arr) / (target_arr + 1e-7))) * 100

        # R-squared (Linear)
        ss_res = np.sum((target_arr - preds_arr) ** 2)
        ss_tot = np.sum((target_arr - np.mean(target_arr)) ** 2)
        r2_lin = 1 - (ss_res / (ss_tot + 1e-7))

        # R-squared (Log)
        ss_res_log = np.sum(log_residuals**2)
        ss_tot_log = np.sum((log_target - np.mean(log_target)) ** 2)
        r2_log = 1 - (ss_res_log / (ss_tot_log + 1e-7))

        # MSLE
        msle = np.mean(log_residuals**2)

        print(f"\nEvaluation Results for NLME Model:")
        print(f"{'Metric':<20} | {'Value':<10}")
        print("-" * 35)
        print(f"{'MAE (Linear)':<20} | {mae_lin:.4e}")
        print(f"{'MAPE (%)':<20} | {mape:.2f}%")
        print(f"{'RMSE (Linear)':<20} | {rmse_lin:.4e}")
        print(f"{'MSLE (Log)':<20} | {msle:.4e}")
        print(f"{'R-squared (Linear)':<20} | {r2_lin:.4f}")
        print(f"{'R-squared (Log)':<20} | {r2_log:.4f}\n")
        # ----------------------------------------------------

        plt.figure(figsize=(6, 6))
        plt.scatter(all_obs, all_pred, alpha=0.5)
        plt.plot([0, max_val], [0, max_val], "r--")
        plt.xlabel("Observed Concentration")
        plt.ylabel("Predicted Concentration")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.grid(True, alpha=0.3)
        plt.title("Observed vs Predicted")
        plt.savefig("misc/nlme_plots/obs_v_pred.png")
        plt.close()


def main(config_path, file_path=None):
    config = getConfig(config_path)
    if not file_path:
        file_path = config.get("data", {}).get("test_file")

    if not file_path:
        print(
            f"Error: No data file found in config {config_path} and no --file provided."
        )
        sys.exit(1)

    if not os.path.exists(file_path):
        # Try relative to root if not found
        alt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_path
        )
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"Error: Data file {file_path} not found.")
            sys.exit(1)

    df, map_estimate, patients, cov_type = run_individual_pipeline(file_path)
    plot_individual_profiles(df, map_estimate, patients, cov_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="config.toml", help="Path to config file"
    )
    parser.add_argument("-f", "--file", help="Path to data file (overrides config)")
    args = parser.parse_args()
    main(args.config, args.file)
