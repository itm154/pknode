# pyright: reportAttributeAccessIssue=false

import pandas as pd
import torch
from torch import Tensor

# Data/Dataset is based on the Monolix Format
# https://monolixsuite.slp-software.com/monolix/2024R1/data-format


class PKData:
    """
    Class to create the data from the file
    """

    def __init__(self, data_file: str, col_mapping: dict):
        self.df = pd.read_csv(data_file)
        self.cols = col_mapping
        self.patients = self.df[self.cols["id"]].unique()

        # Pre-filter to separate observations and administrations
        self.obs_df = self.df[self.df[self.cols["evid"]] == 0]
        self.admin_df = self.df[self.df[self.cols["evid"]] == 1]

    def get_patient_data(self, patient_id) -> dict:
        """
        Extracts time, concentration, doses, and covariates for a single patient
        """
        p_obs = self.obs_df[self.obs_df[self.cols["id"]] == patient_id]
        p_admin = self.admin_df[self.admin_df[self.cols["id"]] == patient_id]

        data = {
            "times": p_obs[self.cols["time"]].values,
            "conc": p_obs[self.cols["conc"]].values,
            "admin_times": p_admin[self.cols["time"]].values,
            "doses": p_admin[self.cols["dose"]].values,
        }

        if "covariates" in self.cols:
            # Assume covariates are static for the patient (take first row)
            pat_row = self.df[self.df[self.cols["id"]] == patient_id].iloc[0]
            data["covariates"] = pat_row[self.cols["covariates"]].values.astype(float)

        return data
