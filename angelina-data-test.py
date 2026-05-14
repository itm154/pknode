# pyright: reportAttributeAccessIssue=false

import pandas as pd


# Data/Dataset is based on the Monolix Format
# https://monolixsuite.slp-software.com/monolix/2024R1/data-format
class PKData:
  
    def __init__(self, data_file: str, col_mapping: dict):
        # Load data
        self.df = pd.read_csv(data_file)

        # Sort data
        self.df = self.df.sort_values(
            by=[col_mapping["id"], col_mapping["time"]]
            ).reset_index(drop=True)

        # Encode categorical covariates
        if "SEX" in self.df.columns:
            self.df["SEX"]= self.df["SEX"].replace({
                "M": 0,
                "F": 1,
                "Male": 0,
                "Female": 1
            })

            self.df["SEX"] = pd.to_numeric(
                self.df["SEX"],
                errors="coerce"
            )

        # Normalize continuous covariates
        continuous_covs = []

        for cov in ["WT", "AGE","CLCR"]:
            if cov in self.df.columns:
                continuous_covs.append(cov)

        # Fill missing continuous covariate
        if len(continuous_covs) > 0:
            self.df[continuous_covs] = (
                self.df[continuous_covs].apply(pd.to_numeric, errors="coerce")
            )

        self.df[continuous_covs] = (
            self.df[continuous_covs]
            .fillna(self.df[continuous_covs].mean())
        )

        # Normalize covariates
        self.norm_stats = {}

        for cov in continuous_covs:
            mean = self.df[cov].mean()
            std = self.df[cov].std()
                
            self.norm_stats[cov] = {"mean": mean, "std": std}

            # Avoid division by zero in case of constant covariate
            if std !=0 and not pd.isna(std):
                self.df[cov] = (self.df[cov] - mean) / std
            else:
                self.df[cov]= 0.0

        self.cols = col_mapping

        # Convert numeric columns safely
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

        # Remove invalid concentration observations
        if "DVID" in self.cols:
            self.obs_df = self.obs_df[self.obs_df["DVID"] == 1]
        
        self.obs_df = self.obs_df.dropna(subset=[self.cols["conc"]])
        self.admin_df = self.df[self.df[self.evid_col] == 1]

    # Get single patient data
    def get_patient_data(self, patient_id) -> dict:
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

            covs = self.cols["covariates"]

            #Ensure covariates is always a list
            if isinstance(covs, str):
                covs = [covs]

            data["covariates"] = (pat_row[covs].values.astype(float))

        return data
