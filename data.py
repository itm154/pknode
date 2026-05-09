# pyright: reportAttributeAccessIssue=false

import pandas as pd


# Data/Dataset is based on the Monolix Format
# https://monolixsuite.slp-software.com/monolix/2024R1/data-format
class PKData:
    """
    Class to create the data from the file
    """

    def __init__(
        self,
        data_file: str,
        col_mapping: dict,
        cov_means: list | None = None,
        cov_stds: list | None = None,
    ):
        self.df = pd.read_csv(data_file)
        self.cols = col_mapping

        # Convert any "." in dataset to NaN
        for col in [
            self.cols.get("dose"),
            self.cols.get("conc"),
            self.cols.get("time"),
        ]:
            if col and col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # If EVID is missing, infer our own
        # If amount > 0 then it is an administration (1)
        # Rows where DV != NaN is an observation
        if "evid" not in self.cols or self.cols["evid"] not in self.df.columns:
            self.df["EVID_INFERRED"] = 0
            self.df.loc[self.df[self.cols["dose"]] > 0, "EVID_INFERRED"] = 1
            self.evid_col = "EVID_INFERRED"
        else:
            self.evid_col = self.cols["evid"]

        self.patients = self.df[self.cols["id"]].unique()
        self.obs_df = self.df[self.df[self.evid_col] == 0]
        self.admin_df = self.df[self.df[self.evid_col] == 1]

        # Scale covariates using z-score (see line 80)
        if "covariates" in self.cols:
            if cov_means is not None and cov_stds is not None:
                self.cov_means = cov_means
                self.cov_stds = cov_stds
            else:
                self.cov_means = self.df[self.cols["covariates"]].mean().values
                self.cov_stds = self.df[self.cols["covariates"]].std().values.copy()
                # Avoid division by zero
                self.cov_stds[self.cov_stds == 0] = 1.0  # pyright: ignore

    def get_patient_data(self, patient_id) -> dict:
        """
        Extracts time, concentration, doses, and covariates for a single patient
        """
        p_obs = self.obs_df[self.obs_df[self.cols["id"]] == patient_id].copy()
        p_admin = self.admin_df[self.admin_df[self.cols["id"]] == patient_id].copy()

        # Handle duplicate observation times (e.g., if PD data is also present)
        # Keep only the first record for each time point to avoid odeint errors
        p_obs = p_obs.drop_duplicates(subset=[self.cols["time"]])  # pyright: ignore

        data = {
            "times": p_obs[self.cols["time"]].values.astype(float),
            "conc": p_obs[self.cols["conc"]].values.astype(float),
            "admin_times": p_admin[self.cols["time"]].values.astype(float),
            "doses": p_admin[self.cols["dose"]].values.astype(float),
        }

        if "covariates" in self.cols:
            # Assume covariates are static for the patient (take first row)
            pat_row = self.df[self.df[self.cols["id"]] == patient_id].iloc[0]
            covs = pat_row[self.cols["covariates"]].values.astype(float)
            # Apply scaling
            data["covariates"] = (covs - self.cov_means) / self.cov_stds

        return data
