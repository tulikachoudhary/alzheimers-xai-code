# make_lime_pdp_tabular.py — PDP/ICE + LIME (HTML & PNG) for tabular AD classifier
# Outputs → C:\Users\TulikaChoudhary\Desktop\c502\VIS\lime_pdp

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from lime.lime_tabular import LimeTabularExplainer
import joblib

# ---------- CONFIG ----------
BASE       = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV_PATH   = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
OUT_DIR    = os.path.join(BASE, "VIS", "lime_pdp")
MODELS_DIR = os.path.join(BASE, "MODELS")
RF_PKL = os.path.join(MODELS_DIR, "rf_tabular.pkl")
LR_PKL = os.path.join(MODELS_DIR, "lr_tabular.pkl")

SEED = 42
TOP_N_FEATURES = 8
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- UTIL ----------
def pick_label_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """Find a DX column and map to binary AD (1) vs non-AD (0)."""
    cand = None
    for c in df.columns:
        if str(c).upper() == "DX_BL":
            cand = c; break
    if cand is None:
        for c in df.columns:
            if str(c).upper().startswith("DX"):
                cand = c; break
    if cand is None:
        raise SystemExit("[FATAL] No DX/DX_bl column found.")

    y_raw = df[cand]
    if pd.api.types.is_numeric_dtype(y_raw):
        vals = pd.unique(pd.to_numeric(y_raw, errors="coerce").dropna())
        if set(vals).issubset({0,1}):
            return cand, y_raw.astype(int)

    def map_dx(v):
        v = str(v).strip().upper()
        if v in {"AD","DEMENTIA","DEMENTED","PROBABLE AD","ALZHEIMERS","ALZHEIMER'S"}: return 1
        if v in {"CN","CTL","CONTROL","NORMAL"}: return 0
        if v in {"EMCI","LMCI","MCI","SMC","MCI-NON"}: return 0
        try: return int(float(v) >= 1.0)
        except: return np.nan

    y = y_raw.map(map_dx).fillna(0).astype(int)
    return cand, y

def pick_feature_matrix(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Keep numeric features; drop known IDs/date/path columns + label."""
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE","MRI","NII"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    drop_cols = [c for c in X.columns if any(k in str(c).upper() for k in ban)]
    if label_col in X.columns: drop_cols.append(label_col)
    X = X.drop(columns=list(set(drop_cols)), errors="ignore")
    if X.shape[1] == 0:
        raise SystemExit("[FATAL] No numeric features remain after filtering.")
    return X

def ensure_models(X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
    """Load existing RF/LR or train fresh with an impute+scale preprocessor."""
    num_cols = list(X.columns)
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler(with_mean=True, with_std=True))])
    pre = ColumnTransformer([("num", numeric_pipe, num_cols)], remainder="drop")

    def make_rf():
        return Pipeline([("pre", pre),
                         ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                        random_state=SEED, n_jobs=-1,
                                                        class_weight="balanced_subsample"))])
    def make_lr():
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                                    class_weight="balanced",
                                                    random_state=SEED, max_iter=200))])

    rf = joblib.load(RF_PKL) if os.path.isfile(RF_PKL) else None
    lr = joblib.load(LR_PKL) if os.path.isfile(LR_PKL) else None
    if rf is None or not isinstance(rf, Pipeline): rf = make_rf().fit(X, y); joblib.dump(rf, RF_PKL)
    if lr is None or not isinstance(lr, Pipeline): lr = make_lr().fit(X, y); joblib.dump(lr, LR_PKL)
    return rf, lr

def eval_and_log(name: str, model: Pipeline, X_test, y_test, out_json: str):
    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    y_pred = (proba >= 0.5).astype(int)
    rep = classification_report(y_test, y_pred, output_dict=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"AUC": float(auc), "report": rep}, f, indent=2)
    print(f"[REPORT] {name}: AUC={auc:.3f}. Saved metrics -> {out_json}")

def top_features_by_rf_importance(rf: Pipeline, feature_names: List[str], k: int) -> List[str]:
    importances = rf.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1][:k]
    return [feature_names[i] for i in idx]

def save_pdp_plots(model: Pipeline, X: pd.DataFrame, feats: List[str], tag: str):
    for feat in feats:
        fig = plt.figure(figsize=(6,4)); ax = plt.gca()
        try:
            PartialDependenceDisplay.from_estimator(
                model, X, [feat], kind="both", target=1, grid_resolution=20, ax=ax
            )
            ax.set_title(f"{tag} PDP/ICE — {feat}")
            outp = os.path.join(OUT_DIR, f"pdp_{tag}_{feat}.png")
            plt.tight_layout(); plt.savefig(outp, dpi=160); plt.close(fig)
            print("[OK] PDP:", outp)
        except Exception as e:
            plt.close(fig); print(f"[WARN] PDP failed for {feat} ({tag}): {e}")

# ---- LIME helper: keep all columns, no NaNs (median; all-NaN → 0) ----
def _median_0_impute_df(df: pd.DataFrame) -> pd.DataFrame:
    med = df.median(numeric_only=True)
    out = df.copy()
    out = out.fillna(med)   # fills columns that have some non-NaN values
    out = out.fillna(0)     # any still-NaN (all-NaN columns) -> 0
    return out

def save_lime_explanations(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           feature_names: List[str], class_names=("non-AD","AD"),
                           n_pos=5, n_neg=5):
    """
    LIME on NaN-free copies, saving BOTH HTML and PNG bar charts for each explained row.
    """
    Xtr_imp_df = _median_0_impute_df(X_train)
    Xte_imp_df = _median_0_impute_df(X_test)

    explainer = LimeTabularExplainer(
        training_data=Xtr_imp_df.values,          # fixed (N, d)
        feature_names=feature_names,              # len d
        class_names=list(class_names),
        discretize_continuous=False,
        random_state=SEED
    )

    # LIME -> model: convert ndarray back to DataFrame with columns
    def predict_fn(A):
        A_df = pd.DataFrame(np.asarray(A), columns=feature_names)
        return model.predict_proba(A_df)

    # pick confident AD-like and CN-like examples by model prob
    proba = model.predict_proba(X_test)[:,1]
    pos_idx = list(np.argsort(proba)[::-1][:n_pos])
    neg_idx = list(np.argsort(proba)[:n_neg])

    def explain_and_save(row_idx: int, label: str):
        x = Xte_imp_df.iloc[row_idx].values  # (d,)
        try:
            exp = explainer.explain_instance(
                data_row=x, predict_fn=predict_fn,
                num_features=min(12, len(feature_names)), labels=[1]
            )
            # Save HTML
            html_path = os.path.join(OUT_DIR, f"lime_{label}_{row_idx:04d}.html")
            exp.save_to_file(html_path)
            print("[OK] LIME HTML:", html_path)

            # Save PNG from exp.as_list(label=1)
            items = exp.as_list(label=1)  # list[(name, weight)]
            names = [t[0] for t in items]
            vals  = [float(t[1]) for t in items]

            order = np.argsort(np.abs(vals))
            names = [names[i] for i in order]
            vals  = [vals[i]  for i in order]

            plt.figure(figsize=(7.5, 5))
            y = np.arange(len(names))
            plt.barh(y, vals)
            plt.yticks(y, names, fontsize=8)
            plt.axvline(0, lw=1)
            plt.title(f"LIME — {label} idx={row_idx} (class=AD)")
            plt.xlabel("Contribution to P(AD)")
            plt.tight_layout()
            png_path = os.path.join(OUT_DIR, f"lime_{label}_{row_idx:04d}.png")
            plt.savefig(png_path, dpi=160)
            plt.close()
            print("[OK] LIME PNG:", png_path)
        except Exception as e:
            print(f"[WARN] LIME failed for {label} idx={row_idx}: {e}")

    for i in pos_idx: explain_and_save(i, "ADlike")
    for i in neg_idx: explain_and_save(i, "CNlike")

# ---------- MAIN ----------
def main():
    warnings.filterwarnings("ignore")
    print("[1/6] Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    print("[2/6] Building labels…")
    label_col, y = pick_label_column(df)

    print("[3/6] Selecting numeric features…")
    X = pick_feature_matrix(df, label_col)
    feature_names = list(X.columns)
    print(f"[INFO] Using {len(feature_names)} numeric features. Examples: {feature_names[:8]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    print("[4/6] Load/train models…")
    rf, lr = ensure_models(X_train, y_train, feature_names)

    print("[5/6] Evaluate & report…")
    eval_and_log("RandomForest", rf, X_test, y_test, os.path.join(OUT_DIR, "rf_metrics.json"))
    eval_and_log("LogReg",       lr, X_test, y_test, os.path.join(OUT_DIR, "lr_metrics.json"))

    top_feats = top_features_by_rf_importance(rf, feature_names, TOP_N_FEATURES)
    print(f"[INFO] Top features for PDP: {top_feats}")

    print("[6/6] PDP/ICE plots…")
    save_pdp_plots(rf, X_test, top_feats, tag="RF")
    save_pdp_plots(lr, X_test, top_feats, tag="LR")

    print("[EXTRA] LIME explanations (RF)…")
    save_lime_explanations(rf, X_train, X_test, feature_names)

    print("[EXTRA] LIME explanations (LR)…")
    save_lime_explanations(lr, X_train, X_test, feature_names)

    manifest = {
        "csv": CSV_PATH,
        "label_col": label_col,
        "n_features": len(feature_names),
        "top_pdp_features": top_feats,
        "models": {"rf": RF_PKL, "lr": LR_PKL},
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n[DONE] Outputs →", OUT_DIR)

if __name__ == "__main__":
    main()
