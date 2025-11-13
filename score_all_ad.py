# score_all_ad.py
import pandas as pd, numpy as np
from pathlib import Path
import joblib

ROOT = Path(r"C:\Users\tulikachoudhary\Desktop\c502")
MERGED = ROOT / "FEATURES" / "merged_features.csv"
MODELS = ROOT / "MODELS"
OUT = ROOT / "FEATURES" / "ad_predictions.csv"

df = pd.read_csv(MERGED)

feat_cols = ['vox_mm3','brain_voxels','ICV_ml',
             'int_mean','int_std','int_median','int_IQR','int_skew','int_kurtosis',
             'CSF_ml','GM_ml','WM_ml','AGE','PTEDUCAT','APOE4']
X = df[feat_cols].copy()
for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

probs = {}
rf_path = MODELS / "ad_rf.joblib"
lr_path = MODELS / "ad_logreg.joblib"

if rf_path.exists():
    rf = joblib.load(rf_path)
    p = rf.predict_proba(X)[:, 1]
    probs["Prob_AD_RF"] = p
else:
    probs["Prob_AD_RF"] = np.nan

if lr_path.exists():
    lr = joblib.load(lr_path)
    p = lr.predict_proba(X)[:, 1]
    probs["Prob_AD_LR"] = p
else:
    probs["Prob_AD_LR"] = np.nan

out = df[["subject","PTID","DX_bl"]].copy()
for k,v in probs.items():
    out[k] = v

out.to_csv(OUT, index=False)
print(f"Saved predictions -> {OUT}")
print(out)
