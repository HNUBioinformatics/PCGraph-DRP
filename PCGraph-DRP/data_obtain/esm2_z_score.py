import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

in_path  = r"../PCGraph-DRP/pathway_network/pathway_protein_esm2_650M_features.csv"
out_path = r"../PCGraph-DRP/pathway_network/pathway_ESM2_features_scaled.csv"
scaler_path = r"../PCGraph-DRP/pathway_network/pathway_esm2_pca_scaler.joblib"

df = pd.read_csv(in_path, index_col=0)

esm2_cols = [c for c in df.columns if c.startswith("pathway_esm2_pca_")]
if not esm2_cols:
    esm2_cols = [c for c in df.columns if c.startswith("esm2_")]

if not esm2_cols:
    raise ValueError("No columns starting with esm2_pca_ (or esm2_) were found, please check the file.")

print(f"Detected {len(esm2_cols)} ESM2 feature columns. Examples: {esm2_cols[:5]} ...")

if df[esm2_cols].isna().sum().sum() > 0:
    df[esm2_cols] = df[esm2_cols].fillna(0.0)

scaler = StandardScaler()
df[esm2_cols] = scaler.fit_transform(df[esm2_cols])

df.to_csv(out_path)
joblib.dump({"esm2_cols": esm2_cols, "scaler": scaler}, scaler_path)

print("Standardized file saved to:", out_path)
print("Scaler saved to:", scaler_path)
print("Verification: Average absolute mean=%.4f  Average std=%.4f" % (
    df[esm2_cols].mean().abs().mean(),
    df[esm2_cols].std().mean()
))