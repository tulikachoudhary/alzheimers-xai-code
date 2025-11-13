# train_ad_binary.py
import warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")  # keep console clean for tiny data

ROOT = Path(r"C:\Users\tulikachoudhary\Desktop\c502")
MERGED = ROOT / "FEATURES" / "merged_features.csv"
OUTDIR = ROOT / "MODELS"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Reading: {MERGED}")
df = pd.read_csv(MERGED)

# Label: AD -> 1, everything else -> 0
y = (df["DX_bl"].astype(str).str.upper() == "AD").astype(int).to_numpy()

# Features used
feat_cols = ['vox_mm3','brain_voxels','ICV_ml',
             'int_mean','int_std','int_median','int_IQR','int_skew','int_kurtosis',
             'CSF_ml','GM_ml','WM_ml','AGE','PTEDUCAT','APOE4']
X = df[feat_cols].copy()

# Minimal imputation for safety
for c in X.columns:
    if X[c].isna().any():
        X[c] = X[c].fillna(X[c].median())

X = X.to_numpy()

# Simple class balance info
unique, counts = np.unique(y, return_counts=True)
print("\nClass counts:", dict(zip(unique, counts)))

# Pipelines
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=500))
])
pipe_rf = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1, random_state=42, class_weight="balanced"
    ))
])

def cv_bal_acc(model, X, y, n_splits=2, n_repeats=10, seed=42):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = []
    total = 0
    skipped = 0
    for train_idx, test_idx in rskf.split(X, y):
        total += 1
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            skipped += 1
            continue  # skip invalid fold (train set single-class)
        model.fit(X[train_idx], y_tr)
        y_pred = model.predict(X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], y_pred))
    return np.array(scores), total, skipped

def report(name, pipe):
    scores, total, skipped = cv_bal_acc(pipe, X, y, n_splits=2, n_repeats=10, seed=42)
    if scores.size == 0:
        print(f"\n{name} — CV Balanced Accuracy")
        print(f"No valid folds (dataset too imbalanced).")
    else:
        print(f"\n{name} (AD vs non-AD) — CV Balanced Accuracy")
        print(f"valid_folds={len(scores)}  skipped={skipped}/{total}  mean={np.nanmean(scores):.3f}  std={np.nanstd(scores):.3f}")
        print(f"all={np.round(scores,3).tolist()}")

report("LogReg", pipe_lr)
report("RF   ", pipe_rf)

# Fit on ALL data to export models
# Note: for LR, fitting on all data still requires both classes. Guard it.
if len(np.unique(y)) > 1:
    pipe_lr.fit(X, y)
    import joblib
    joblib.dump(pipe_lr, OUTDIR / "ad_logreg.joblib")
else:
    print("\n[Info] Skipping LogReg export: only one class present in full data.")

import joblib
pipe_rf.fit(X, y)
joblib.dump(pipe_rf, OUTDIR / "ad_rf.joblib")

print(f"\nModels saved to: {OUTDIR}")
print("Done.")
