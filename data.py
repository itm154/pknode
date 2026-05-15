from typing import Any, Optional

import numpy as np
import pandas as pd


# Data/Dataset is based on the Monolix Format
# https://monolixsuite.slp-software.com/monolix/2024R1/data-format
class PKData:
    """
    Class to create the data from the file
    """

    df: pd.DataFrame
    obs_df: pd.DataFrame
    admin_df: pd.DataFrame
    cov_means: np.ndarray
    cov_stds: np.ndarray

    def __init__(
        self,
        data_file: str,
        col_mapping: dict[str, str],
        cov_means: Optional[np.ndarray] = None,
        cov_stds: Optional[np.ndarray] = None,
    ):
        self.df = pd.read_csv(data_file)
        self.cols = col_mapping

        # Convert any "." in dataset to NaN
        for col_key in ["dose", "conc", "time"]:
            col = self.cols.get(col_key)
            if col and col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # If EVID is missing, infer our own
        # If amount > 0 then it is an administration (1)
        # Rows where DV != NaN is an observation
        if "evid" not in self.cols or self.cols["evid"] not in self.df.columns:
            self.df["EVID_INFERRED"] = 0
            dose_col = self.cols["dose"]
            self.df.loc[self.df[dose_col] > 0, "EVID_INFERRED"] = 1
            self.evid_col = "EVID_INFERRED"
        else:
            self.evid_col = self.cols["evid"]

        self.patients = self.df[self.cols["id"]].unique()
        self.obs_df = self.df[self.df[self.evid_col] == 0]
        self.admin_df = self.df[self.df[self.evid_col] == 1]

        # Scale covariates using z-score (see line 80)
        if "covariates" in self.cols:
            cov_cols = self.cols["covariates"]
            if cov_means is not None and cov_stds is not None:
                self.cov_means = cov_means
                self.cov_stds = cov_stds
            else:
                cov_data = self.df[cov_cols]
                self.cov_means = cov_data.mean().values
                self.cov_stds = cov_data.std().values.copy()
                # Avoid division by zero
                self.cov_stds[self.cov_stds == 0] = 1.0

    def get_patient_data(self, patient_id: Any) -> dict[str, Any]:
        """
        Extracts time, concentration, doses, and covariates for a single patient
        """
        p_obs = self.obs_df[self.obs_df[self.cols["id"]] == patient_id].copy()
        p_admin = self.admin_df[self.admin_df[self.cols["id"]] == patient_id].copy()

        # Handle duplicate observation times (e.g., if PD data is also present)
        # Keep only the first record for each time point to avoid odeint errors
        p_obs = p_obs.drop_duplicates(subset=[self.cols["time"]])

        data: dict[str, Any] = {
            "times": p_obs[self.cols["time"]].values.astype(float),
            "conc": p_obs[self.cols["conc"]].values.astype(float),
            "admin_times": p_admin[self.cols["time"]].values.astype(float),
            "doses": p_admin[self.cols["dose"]].values.astype(float),
        }

        if "covariates" in self.cols:
            # Assume covariates are static for the patient (take first row)
            pat_df = self.df[self.df[self.cols["id"]] == patient_id]
            if not pat_df.empty:
                pat_row = pat_df.iloc[0]
                data["covariates"] = pat_row[self.cols["covariates"]].values.astype(
                    float
                )

        return data
