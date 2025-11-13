#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, balanced_accuracy_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MERGED = HERE / "FEATURES" / "merged_features.csv"
OUTDIR = HERE / "MODELS"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Reading: {MERGED}")
df = pd.read_csv(MERGED)

# Columns that are definitely not features
non_feature_cols = {
    'subject','path','PTID','RID','VISCODE','VISCODE2',
    'EXAMDATE_bl','EXAMDATE','ORIGPROT','COLPROT','PTGENDER'
}

label_col = "DX_bl"
if label_col not in df.columns:
    raise SystemExit("DX_bl not found in merged_features.csv")

# Drop rows with missing label
df = df.dropna(subset=[label_col]).copy()

# Keep numeric columns as features (excluding helper IDs etc.)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in non_feature_cols]

print("\n=== Columns ===")
print("Features:", feature_cols)
print("Label   :", label_col)

# Encode label
y_text = df[label_col].astype(str).values
classes = sorted(pd.unique(y_text))
class_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
y = np.array([class_to_idx[v] for v in y_text], dtype=int)

X = df[feature_cols].values

# Show class balance
print("\n=== Class balance (all) ===")
print(pd.Series(y_text).value_counts())

# Decide training mode based on smallest class size
counts = pd.Series(y).value_counts()
min_count = counts.min()
n_samples = len(y)

use_stratified = min_count >= 2
can_holdout = use_stratified and n_samples >= 6  # tiny rule of thumb

# Pipelines (impute -> scale -> classifier)
logreg_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, multi_class="ovr",
                               class_weight="balanced", random_state=42))
])

rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced",
        random_state=42, n_jobs=-1
    ))
])

def save_cv_scores(name, scores, fname):
    msg = f"{name} — CV Balanced Accuracy\n" \
          f"n_splits={len(scores)}  mean={scores.mean():.3f}  std={scores.std():.3f}\n" \
          f"all={np.round(scores, 3).tolist()}\n"
    print("\n" + msg.strip())
    (OUTDIR / fname).write_text(msg, encoding="utf-8")

def eval_and_plot_test(name, model, X_test, y_test, idx_to_class):
    y_pred = model.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred,
        target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
        digits=3
    )
    # Save text report
    rep_path = OUTDIR / f"{name}_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"{name} — Test results\n")
        f.write(f"Balanced Accuracy: {bal_acc:.3f}\n")
        f.write(f"Macro F1        : {f1_macro:.3f}\n\n")
        f.write(report)
    print(f"\n{name} — Test BalancedAcc={bal_acc:.3f}  MacroF1={f1_macro:.3f}")
    print(f"Classification report saved -> {rep_path}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(idx_to_class))))
    fig = plt.figure()
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(f'{name} Confusion Matrix (test)')
    plt.colorbar(im)
    tick_marks = np.arange(len(idx_to_class))
    plt.xticks(tick_marks, [idx_to_class[i] for i in tick_marks], rotation=45, ha='right')
    plt.yticks(tick_marks, [idx_to_class[i] for i in tick_marks])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel('Predicted'); plt.ylabel('True')
    fig.tight_layout()
    figpath = OUTDIR / f"{name}_confusion_matrix.png"
    plt.savefig(figpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved -> {figpath}")

# ====== TRAINING ======
if use_stratified:
    # Stratified CV (5-fold or up to min_count)
    n_splits = min(5, int(min_count))  # safe upper bound
    n_splits = max(n_splits, 2)        # at least 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # CV on all data (report)
    logreg_cv = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    rf_cv = cross_val_score(rf_pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    save_cv_scores("LogReg", logreg_cv, "logreg_cv.txt")
    save_cv_scores("RF    ", rf_cv, "rf_cv.txt")

    if can_holdout:
        # Stratified hold-out
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        logreg_pipe.fit(X_train, y_train)
        rf_pipe.fit(X_train, y_train)
        eval_and_plot_test("logreg", logreg_pipe, X_test, y_test, idx_to_class)
        eval_and_plot_test("rf", rf_pipe, X_test, y_test, idx_to_class)
    else:
        print("\n[Info] Dataset is very small; skipping hold-out test.")

    # Fit final models on ALL data
    logreg_pipe.fit(X, y)
    rf_pipe.fit(X, y)

else:
    # Not enough samples in at least one class for stratification.
    print("\n[Warning] At least one class has only 1 sample. "
          "Using plain K-Fold CV (no stratify) and training on ALL data.")
    n_splits = min(3, n_samples)  # with 7 samples, this is 3
    n_splits = max(n_splits, 2)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    logreg_cv = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    rf_cv = cross_val_score(rf_pipe, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    save_cv_scores("LogReg", logreg_cv, "logreg_cv.txt")
    save_cv_scores("RF    ", rf_cv, "rf_cv.txt")

    # Fit final models on ALL data
    logreg_pipe.fit(X, y)
    rf_pipe.fit(X, y)

# Save models and metadata
dump(logreg_pipe, OUTDIR / "logreg.joblib")
dump(rf_pipe, OUTDIR / "rf.joblib")
(HERE / "MODELS" / "meta.json").write_text(
    json.dumps({
        "feature_cols": feature_cols,
        "classes": classes,
        "class_to_idx": {k:int(v) for k,v in class_to_idx.items()}
    }, indent=2),
    encoding="utf-8"
)
print(f"\nModels saved to: {OUTDIR}")
print("Done.")
