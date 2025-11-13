# retrain_fb.py — retrain from clinician-validated feedback (robust checks)

import os, sqlite3, json, warnings
import numpy as np
import pandas as pd

from typing import List, Tuple
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

# ----------------- PATHS -----------------
BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
DB_PATH = os.path.join(BASE, "hitl.db")
CSV_PATH = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
MODELS_DIR = os.path.join(BASE, "MODELS")
RF_OUT = os.path.join(MODELS_DIR, "rf_tabular_v2.pkl")
LR_OUT = os.path.join(MODELS_DIR, "lr_tabular_v2.pkl")
OUT_CARD = os.path.join(MODELS_DIR, "model_card_v2.json")
MIN_ROWS_TO_SAVE = 40

# ----------------- HELPERS -----------------
def load_feedback_table(db_path: str) -> pd.DataFrame:
    if not os.path.isfile(db_path):
        print(f"[WARN] DB not found: {db_path}")
        return pd.DataFrame()
    con = sqlite3.connect(db_path)
    q = """
    SELECT f.*, p.ptid, p.rid, p.scan_date, p.model_id, p.data_version
    FROM feedback f
    JOIN predictions p ON p.id=f.prediction_id
    WHERE f.use_for_retrain=1 AND f.confidence>=0.7
    """
    df = pd.read_sql_query(q, con)
    con.close()
    return df

def pick_feature_matrix(df: pd.DataFrame, label_col: str = None) -> pd.DataFrame:
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE","MRI","NII"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    drop_cols = [c for c in X.columns if any(k in str(c).upper() for k in ban)]
    if label_col and label_col in X.columns:
        drop_cols.append(label_col)
    X = X.drop(columns=list(set(drop_cols)), errors="ignore")
    if X.shape[1] == 0:
        raise SystemExit("[FATAL] No numeric features remain after filtering.")
    return X

def map_final_label_to_binary(s: pd.Series) -> np.ndarray:
    return s.map({"AD":1,"MCI":0,"CN":0,"OTHER":0}).astype(int).values

def build_preprocessor(num_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

def safe_proba(clf, Xt) -> np.ndarray:
    """Return P(AD) even if estimator was fit on a single class."""
    proba = clf.predict_proba(Xt)
    classes = getattr(clf, "classes_", np.array([0,1]))
    if proba.shape[1] == 2:
        # make sure column aligns to AD=1
        ad_col = list(classes).index(1) if 1 in classes else 1
        return proba[:, ad_col]
    # single-class case: warn and synthesize probs
    only = int(classes[0]) if len(classes) else 0
    print("[WARN] Estimator saw one class only during fit; producing degenerate probabilities.")
    return np.ones(len(Xt)) if only == 1 else np.zeros(len(Xt))

# ----------------- MAIN -----------------
def main():
    print("[1/6] Loading clinician feedback…")
    fb = load_feedback_table(DB_PATH)
    if fb.empty:
        print("[WARN] No approved feedback yet (use_for_retrain=1, confidence>=0.7).")
        return
    print(f"[INFO] Feedback rows total: {len(fb)}")

    print("[2/6] Loading features CSV…")
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"[FATAL] Features CSV not found: {CSV_PATH}")
    feats_df = pd.read_csv(CSV_PATH)
    feats_df["PTID"] = feats_df["PTID"].astype(str)
    fb["ptid"] = fb["ptid"].astype(str)

    print("[3/6] Join feedback → features by PTID…")
    merged = feats_df.merge(fb[["ptid","final_label"]], left_on="PTID", right_on="ptid", how="inner")
    if merged.empty:
        print("[WARN] No PTID overlap; nothing to train.")
        return

    y = map_final_label_to_binary(merged["final_label"])
    X = pick_feature_matrix(merged, label_col=None)
    groups = merged["PTID"].values
    uniq_subjects = pd.Series(groups).nunique()
    class_counts = pd.Series(y).value_counts().to_dict()
    n_pos = class_counts.get(1, 0); n_neg = class_counts.get(0, 0)

    print(f"[INFO] Using {len(X)} rows × {X.shape[1]} features; subjects={uniq_subjects}; class dist={{AD:{n_pos}, nonAD:{n_neg}}}")

    # Hard-stop guards
    if n_pos == 0 or n_neg == 0:
        print("[STOP] Need feedback from at least TWO classes (some AD and some non-AD).")
        print("       Please log a few more clinician reviews including the other class, then rerun.")
        return

    # Build preprocessors
    feats = list(X.columns)
    pre = build_preprocessor(feats)

    print("[4/6] Evaluation…")
    # If we have ≥3 subjects → GroupKFold; else do stratified holdout
    if uniq_subjects >= 3:
        gkf = GroupKFold(n_splits=min(5, uniq_subjects))
        oof_rf, oof_lr, oof_y = [], [], []
        for tr, va in gkf.split(X, y, groups):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]

            # RF
            rf = Pipeline([("pre", pre),
                           ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                          random_state=42, n_jobs=-1,
                                                          class_weight="balanced_subsample"))])
            rf.fit(Xtr, ytr)
            prf = rf.predict_proba(Xva)[:, 1]  # 2 classes guaranteed by earlier check

            # LR + isotonic calibration
            pre_lr = build_preprocessor(feats)
            Xtr_t = pre_lr.fit_transform(Xtr, ytr)
            Xva_t = pre_lr.transform(Xva)
            lr_raw = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                        class_weight="balanced", max_iter=400, random_state=42)
            lr_cal = CalibratedClassifierCV(lr_raw, method="isotonic", cv=3)
            lr_cal.fit(Xtr_t, ytr)
            plr = safe_proba(lr_cal, Xva_t)

            oof_rf.extend(prf); oof_lr.extend(plr); oof_y.extend(yva)

        oof_y = np.array(oof_y); oof_rf = np.array(oof_rf); oof_lr = np.array(oof_lr)
        auc_rf = roc_auc_score(oof_y, oof_rf); brier_rf = brier_score_loss(oof_y, oof_rf)
        auc_lr = roc_auc_score(oof_y, oof_lr); brier_lr = brier_score_loss(oof_y, oof_lr)
        print(f"[CV] RF  : AUC={auc_rf:.3f}  Brier={brier_rf:.4f}")
        print(f"[CV] LR* : AUC={auc_lr:.3f}  Brier={brier_lr:.4f}  (*isotonic calibrated)")
        metrics = {"auc_rf": float(auc_rf), "brier_rf": float(brier_rf),
                   "auc_lr": float(auc_lr), "brier_lr": float(brier_lr)}
    else:
        print("[WARN] Only one/two subjects → using stratified 80/20 holdout.")
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        rf = Pipeline([("pre", pre),
                       ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                      random_state=42, n_jobs=-1,
                                                      class_weight="balanced_subsample"))])
        rf.fit(Xtr, ytr)
        prf = rf.predict_proba(Xva)[:, 1]

        pre_lr = build_preprocessor(feats)
        Xtr_t = pre_lr.fit_transform(Xtr, ytr)
        Xva_t = pre_lr.transform(Xva)
        lr_raw = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                    class_weight="balanced", max_iter=400, random_state=42)
        lr_cal = CalibratedClassifierCV(lr_raw, method="isotonic", cv=3)
        lr_cal.fit(Xtr_t, ytr)
        plr = safe_proba(lr_cal, Xva_t)

        auc_rf = roc_auc_score(yva, prf); brier_rf = brier_score_loss(yva, prf)
        auc_lr = roc_auc_score(yva, plr); brier_lr = brier_score_loss(yva, plr)
        print(f"[HOLDOUT] RF  : AUC={auc_rf:.3f}  Brier={brier_rf:.4f}")
        print(f"[HOLDOUT] LR* : AUC={auc_lr:.3f}  Brier={brier_lr:.4f}  (*isotonic calibrated)")
        metrics = {"auc_rf": float(auc_rf), "brier_rf": float(brier_rf),
                   "auc_lr": float(auc_lr), "brier_lr": float(brier_lr)}

    print("[5/6] Full-fit & optionally save…")
    if len(X) < MIN_ROWS_TO_SAVE:
        print(f"[INFO] Not saving v2 models yet (need ≥{MIN_ROWS_TO_SAVE}, have {len(X)}).")
        print("[DONE] retrain_fb complete.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Full-fit RF
    rf_full = Pipeline([("pre", pre),
                        ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                       random_state=42, n_jobs=-1,
                                                       class_weight="balanced_subsample"))])
    rf_full.fit(X, y)

    # Full-fit LR + calibration
    pre_lr = build_preprocessor(feats)
    Xt = pre_lr.fit_transform(X, y)
    lr_raw = LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                                class_weight="balanced", max_iter=500, random_state=42)
    lr_cal = CalibratedClassifierCV(lr_raw, method="isotonic", cv=5)
    lr_cal.fit(Xt, y)

    import joblib
    joblib.dump(rf_full, RF_OUT)
    joblib.dump({"pre": pre_lr, "cal": lr_cal}, LR_OUT)
    with open(OUT_CARD, "w", encoding="utf-8") as f:
        json.dump({"notes":"v2 trained on human-validated shard","metrics":metrics,
                   "n_rows":int(len(X)),"features":feats}, f, indent=2)
    print(f"[OK] Saved RF v2 → {RF_OUT}")
    print(f"[OK] Saved LR v2 → {LR_OUT}")
    print(f"[OK] Model card   → {OUT_CARD}")
    print("[DONE] retrain_fb complete.")

if __name__ == "__main__":
    main()
