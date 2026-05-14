from data import PKData
import numpy as np

col_mapping = {
    "id": "ID",
    "time": "TIME",
    "dose": "AMT",
    "conc": "DV",
    "covariates": ["WT", "AGE", "SEX"]
}

data = PKData(
    "data/warfarin_data.csv",
    col_mapping
)

# ===== Normalize AGE using z-score =====
age_mean = data.df["AGE"].mean()
age_std = data.df["AGE"].std()

data.df["AGE_ZSCORE"] = (
    data.df["AGE"] - age_mean
) / age_std

# ===== Patient Example =====
patient_id = 100
patient = data.get_patient_data(patient_id)

# Get normalized age
normalized_age = data.df[
    data.df["ID"] == patient_id
]["AGE_ZSCORE"].iloc[0]

print(f"\n=== Patient {patient_id} PK Profile ===")

print(f"\nDose:")
print(f"  {patient['doses'][0]} mg")

print("\nCovariates:")
print(f"  Weight        : {patient['covariates'][0]} kg")
print(f"  Age (z-score) : {normalized_age:.3f}")
print(f"  Sex           : {patient['covariates'][2]}")

print("\nTime-Concentration Profile:")
print("-" * 45)

for t, c in zip(patient['times'], patient['conc']):
    print(f"Time: {t:>6} hr    Concentration: {c:>6}")