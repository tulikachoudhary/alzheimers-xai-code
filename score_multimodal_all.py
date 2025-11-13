# score_all_ad.py — score tabular AD risk and register cases for clinician feedback
# Outputs:
#   - FEATURES\ad_predictions.csv
# Requires:
#   - MODELS\rf_tabular.pkl  (or lr_tabular.pkl if you switch MODEL_PKL)
#   - FEATURES\features_with_genetic_rowwise.csv
# Optional (recommended):
#   - FastAPI server running: uvicorn api_feedback:app --reload

import os, sys, json, warnings
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import requests

# ---------- CONFIG ----------
BASE        = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV_PATH    = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
OUT_CSV     = os.path.join(BASE, "FEATURES", "ad_predictions.csv")
MODELS_DIR  = os.path.join(BASE, "MODELS")
# Choose model: "rf_tabular.pkl" or "lr_tabular.pkl"
MODEL_PKL   = os.path.join(MODELS_DIR, "rf_tabular.pkl")

API_BASE    = "http://127.0.0.1:8000"   # feedback API (optional)
MODEL_ID    = "RF_tabular_v1"           # shown to clinicians
DATA_VERSION= "features_with_genetic_rowwise.csv"

# register toggle via CLI flag
REGISTER = True
if "--no-register" in sys.argv:
    REGISTER = False

warnings.filterwarnings("ignore")


# ---------- HELPERS ----------
def pick_feature_matrix(df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
    """
    Keep numeric features and drop known identifiers / date-ish / path-ish columns.
    Must match the logic used in make_lime_pdp_tabular.py so preprocessing is consistent.
    """
    ban_keys = ["DX", "DIAG", "PATH", "PTID", "RID", "SUBJECT", "SCANDATE", "EXAMDATE",
                "VISCODE", "MRI", "NII"]
    X = df.select_dtypes(include=["number", "bool"]).copy()
    # drop any column whose name contains a banned token (case-insensitive)
    drop_cols = [c for c in X.columns if any(k in str(c).upper() for k in ban_keys)]
    if label_col and label_col in X.columns:
        drop_cols.append(label_col)
    X = X.drop(columns=list(set(drop_cols)), errors="ignore")
    if X.shape[1] == 0:
        raise SystemExit("[FATAL] After filtering, no numeric features remain. Revisit ban_keys or CSV.")
    return X


def safe_get(row, key, default=""):
    val = row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)
    return "" if pd.isna(val) else val


def load_model(path: str) -> Pipeline:
    if not os.path.isfile(path):
        raise SystemExit(f"[FATAL] Missing model file: {path}")
    mdl = joblib.load(path)
    if not isinstance(mdl, Pipeline):
        raise SystemExit("[FATAL] Loaded model is not a sklearn Pipeline. "
                         "Use rf_tabular.pkl/lr_tabular.pkl trained by make_lime_pdp_tabular.py.")
    return mdl


def register_prediction(row: pd.Series, prob_ad: float, yhat_label: str,
                        extra_artifacts: Optional[dict] = None) -> Optional[int]:
    """
    Send one prediction entry to the feedback API.
    If API is offline, print a warning and continue.
    """
    if not REGISTER:
        return None

    # generic artifacts you already have (global PDPs); you can extend with per-subject LIME later
    artifacts = {
        "pdp_imgs": [
            os.path.join("VIS", "lime_pdp", "pdp_RF_CDRSB.png"),
            os.path.join("VIS", "lime_pdp", "pdp_RF_MMSE.png"),
        ],
        "lime_imgs": []
    }
    if extra_artifacts:
        artifacts.update(extra_artifacts)

    payload = {
        "ptid": str(safe_get(row, "PTID", "unknown")),
        "rid": str(safe_get(row, "RID", "unknown")),
        "scan_date": str(safe_get(row, "ScanDate", "")),
        "y_hat_prob": float(prob_ad),
        "y_hat_label": str(yhat_label),
        "model_id": MODEL_ID,
        "data_version": DATA_VERSION,
        "artifacts": artifacts
    }

    try:
        r = requests.post(f"{API_BASE}/register_prediction", json=payload, timeout=5)
        r.raise_for_status()
        pid = r.json().get("prediction_id")
        print(f"[REG] prediction_id={pid} | PTID={payload['ptid']} | prob={prob_ad:.3f}")
        return int(pid) if pid is not None else None
    except Exception as e:
        print(f"[WARN] Could not register prediction (API offline?): {e}")
        return None


# ---------- MAIN ----------
def main():
    print("[1/4] Loading CSV:", CSV_PATH)
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"[FATAL] CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # (Optional) detect a label col, but scoring does not require label. We keep for compatibility.
    label_col = None
    for c in df.columns:
        if str(c).upper() == "DX_BL":
            label_col = c; break
    if label_col is None:
        for c in df.columns:
            if str(c).upper().startswith("DX"):
                label_col = c; break

    print("[2/4] Selecting numeric features…")
    X = pick_feature_matrix(df, label_col=label_col)
    feature_names = list(X.columns)
    print(f"[INFO] Using {len(feature_names)} features. Examples: {feature_names[:8]}")

    print("[3/4] Loading model:", MODEL_PKL)
    model = load_model(MODEL_PKL)
    # sanity: model should expose predict_proba
    if not hasattr(model, "predict_proba"):
        raise SystemExit("[FATAL] Model pipeline has no predict_proba().")

    print("[3.1/4] Scoring…")
    proba_ad = model.predict_proba(X)[:, 1]
    yhat = (proba_ad >= 0.5).astype(int)
    yhat_label = np.where(yhat == 1, "AD", "CN")

    # build output table
    print("[3.2/4] Building output dataframe…")
    out = pd.DataFrame({
        "PTID": df.get("PTID"),
        "RID": df.get("RID"),
        "ScanDate": df.get("ScanDate"),
        "Prob_AD": proba_ad,
        "Pred_Label": yhat_label
    })

    # carry a subject/filename field if you have one
    # try common columns users have in prior scripts
    for col_try in ["subject", "SUBJECT", "MRI_PATH"]:
        if col_try in df.columns:
            out[col_try] = df[col_try]
            break

    # write predictions
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved predictions → {OUT_CSV}  (rows={len(out)})")

    # Register each prediction to the feedback API (optional)
    if REGISTER:
        print("[4/4] Registering predictions to feedback API…")
        for i, row in out.iterrows():
            pid = register_prediction(row, float(row["Prob_AD"]), str(row["Pred_Label"]))
        print("[DONE] Registration pass complete.")
    else:
        print("[4/4] Skipped registration (use --no-register to skip, default is ON).")

    print("[DONE] Scoring complete.")


if __name__ == "__main__":
    main()
