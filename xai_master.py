# xai_master.py  —  one-click XAI report (tabular SHAP/LIME/PDP + MRI figs)
# Paths assume your current project layout on Windows.
# ----------------------------------------------------

import os, json, shutil, warnings
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay

# Optional deps (we gate usage gracefully)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False


# -----------------------
# CONFIG
# -----------------------
BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
PRED = os.path.join(BASE, "FEATURES", "topK_confident_AD.csv")  # optional, if present
MM_PRED = os.path.join(BASE, "FEATURES", "mm_predictions.csv")   # optional fallback
VIS = os.path.join(BASE, "VIS")
OUT = os.path.join(VIS, "XAI_Report")
TAB_SHAP_DIR = os.path.join(OUT, "tabular_shap")
TAB_LIME_DIR = os.path.join(OUT, "tabular_lime")
TAB_PDP_DIR  = os.path.join(OUT, "tabular_pdp")
MRI_DIR      = os.path.join(OUT, "mri")
GRADCAM_DIR  = os.path.join(VIS, "gradcam")  # produced by gradcam_3d.py
SLICES_SRC   = os.path.join(VIS)              # mid-slice pngs live directly under VIS (e.g., 000_PTID_ProbX.png)

TOPK = 25     # how many top MRI figures to collect
SEED = 42     # reproducibility
MAX_LIME = 40 # how many LIME row plots to export
warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs(OUT, exist_ok=True)


# -----------------------
# Utilities
# -----------------------
def prepare_tabular(df):
    """Return (X_train, X_test, y_train, y_test, feat_names, scaler, dx_col, X_train_df, X_test_df)"""
    # Auto-pick diagnosis column
    dx_col = next((c for c in df.columns if str(c).upper() == "DX_BL"), None)
    if dx_col is None:
        dx_col = next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx_col is None:
        raise SystemExit("[FATAL] Need a diagnosis column (DX_BL or DX*) in the merged CSV.")

    # Keep AD/MCI/CN only, build label 1(AD)/0(other)
    keep = df[dx_col].astype(str).str.upper().isin({"AD", "MCI", "CN"})
    df = df.loc[keep].copy()
    y = (df[dx_col].astype(str).str.upper() == "AD").astype(int).values

    # Build numeric feature matrix; drop obvious leakage/id cols if present
    ban = ["DX", "DIAG", "PATH", "PTID", "RID", "SUBJECT", "SCANDATE", "EXAMDATE", "VISCODE"]
    Xn = df.select_dtypes(include=["number", "bool"]).copy()
    drop_cols = [c for c in Xn.columns if any(k in c.upper() for k in ban)]
    Xn = Xn.drop(columns=drop_cols, errors="ignore").fillna(0).astype(np.float32)

    feat_names = list(Xn.columns)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xn)

    # Split (row-level stratified — OK for explainability demos)
    strat = y if len(np.unique(y)) > 1 else None
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=strat
    )

    X_train, X_test = Xs[idx_tr], Xs[idx_te]
    y_train, y_test = y[idx_tr], y[idx_te]
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feat_names)

    return X_train, X_test, y_train, y_test, feat_names, scaler, dx_col, X_train_df, X_test_df


def train_tabular_model(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


# -----------------------
# SHAP (fixed & robust)
# -----------------------
def shap_global_and_rowplots(model, X_train, X_test, y_test, feat_names, out_dir,
                             top_k_dependence=10, max_force=50):
    os.makedirs(out_dir, exist_ok=True)
    dep_dir   = os.path.join(out_dir, "dependence")
    water_dir = os.path.join(out_dir, "waterfall")
    os.makedirs(dep_dir, exist_ok=True)
    os.makedirs(water_dir, exist_ok=True)

    if not _HAS_SHAP:
        print("[SKIP] SHAP not installed. Run: pip install shap")
        return

    # Keep names attached to matrices
    feat_names = list(feat_names)
    X_train_df = pd.DataFrame(X_train, columns=feat_names)
    X_test_df  = pd.DataFrame(X_test,  columns=feat_names)

    # Explainer
    explainer = shap.TreeExplainer(model)
    try:
        shap_values = explainer.shap_values(X_test_df)
    except Exception:
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_test_df)

    # Binary: positive class
    sv_pos = shap_values[1] if isinstance(shap_values, (list, tuple)) and len(shap_values) > 1 else np.array(shap_values)

    # --- HARD ALIGNMENT (fixes "x and y must be same size") ---
    if sv_pos.ndim == 1:
        sv_pos = sv_pos.reshape(1, -1)
    n = min(sv_pos.shape[0], X_test_df.shape[0])
    sv_pos = sv_pos[:n]
    X_test_df = X_test_df.iloc[:n, :]

    if sv_pos.shape[1] != len(feat_names):
        m = min(sv_pos.shape[1], len(feat_names))
        sv_pos = sv_pos[:, :m]
        X_test_df = X_test_df.iloc[:, :m]
        feat_names = feat_names[:m]

    # Global bar
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv_pos, X_test_df, feature_names=feat_names, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "global_bar.png"), dpi=150); plt.close()

    # Global beeswarm
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv_pos, X_test_df, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "global_beeswarm.png"), dpi=150); plt.close()

    # Dependence plots via integer indices
    mean_abs = np.mean(np.abs(sv_pos), axis=0)
    order = np.argsort(-mean_abs)[:min(top_k_dependence, len(feat_names))]
    order = [int(i) for i in np.ravel(order)]
    for idx_i in order:
        name = feat_names[idx_i]
        try:
            plt.figure(figsize=(6, 4))
            shap.dependence_plot(
                idx_i, sv_pos, X_test_df,
                feature_names=feat_names,
                interaction_index=None,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(dep_dir, f"dependence_{idx_i:03d}_{name}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] SHAP dependence failed for {name}: {e}")

    # Per-row WATERFALL plots
    try:
        base_val = np.mean(explainer.expected_value) if isinstance(explainer.expected_value, (list, tuple, np.ndarray)) \
            else float(explainer.expected_value)
    except Exception:
        base_val = 0.0

    rows = min(max_force, X_test_df.shape[0])
    for i in range(rows):
        try:
            exp = shap.Explanation(
                values=sv_pos[i],
                base_values=base_val,
                data=X_test_df.iloc[i].values,
                feature_names=feat_names
            )
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(water_dir, f"waterfall_row_{i:04d}.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[WARN] SHAP waterfall failed for row {i}: {e}")

    print(f"[OK] Tabular SHAP written -> {out_dir}")


# -----------------------
# PDP (with DataFrame & names)
# -----------------------
def run_pdp(model, X_train, feat_names, out_dir, top_k=8):
    os.makedirs(out_dir, exist_ok=True)
    X_train_df = pd.DataFrame(X_train, columns=feat_names)

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feat_names), dtype=np.float32)

    idx = np.argsort(-importances)[:min(top_k, len(feat_names))]
    idx = [int(i) for i in idx]
    sel_features = [feat_names[i] for i in idx]

    for name in sel_features:
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            PartialDependenceDisplay.from_estimator(
                model, X_train_df, [name], ax=ax, kind="average"
            )
            ax.set_title(f"PDP: {name}")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"pdp_{name}.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] PDP failed for {name}: {e}")
    print(f"[OK] Tabular PDP written -> {out_dir}")


# -----------------------
# LIME (tabular)
# -----------------------
def run_lime(model, X_train, X_test, feat_names, out_dir, max_rows=40):
    os.makedirs(out_dir, exist_ok=True)
    if not _HAS_LIME:
        print("[SKIP] LIME not installed. Run: pip install lime")
        return

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feat_names,
        class_names=["Non-AD", "AD"],
        discretize_continuous=True,
        random_state=SEED,
        verbose=False
    )
    # LIME expects predict_proba
    def predictor(X):
        return model.predict_proba(X)

    rows = min(max_rows, X_test.shape[0])
    for i in range(rows):
        try:
            exp = explainer.explain_instance(X_test[i], predictor, num_features=min(10, len(feat_names)))
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(7, 5)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"lime_row_{i:04d}.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] LIME failed for row {i}: {e}")

    print(f"[OK] Tabular LIME written -> {out_dir}")


# -----------------------
# MRI figure gather (Grad-CAM + mid-slices)
# -----------------------
def gather_mri_figs(out_dir, limit=TOPK):
    os.makedirs(out_dir, exist_ok=True)

    # Grad-CAM overlays
    grad_list = sorted(glob(os.path.join(GRADCAM_DIR, "cam_*.png")))

    # Mid-slice montages: any *_Prob*.png in VIS/
    slice_list = sorted(glob(os.path.join(SLICES_SRC, "*_Prob*.png")))

    gcount = 0
    for p in grad_list[:limit]:
        shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
        gcount += 1

    scount = 0
    for p in slice_list[:limit]:
        bn = os.path.basename(p)
        if not os.path.exists(os.path.join(out_dir, bn)):
            shutil.copy2(p, os.path.join(out_dir, bn))
            scount += 1

    print(f"[OK] MRI figures gathered -> {out_dir}")
    print(f"     Grad-CAM overlays: {gcount}")
    print(f"     Mid-slice montages: {scount}")
    return gcount, scount


# -----------------------
# HTML report builder
# -----------------------
def build_html_report(out_dir):
    html = []
    add = html.append

    def section(title):
        add(f"<h2>{title}</h2>")

    def img_grid(title, folder, exts=(".png", ".jpg", ".jpeg")):
        section(title)
        files = []
        for e in exts:
            files += sorted(glob(os.path.join(folder, f"*{e}")))
        if not files:
            add("<p><em>No figures found.</em></p>")
            return
        add('<div style="display:flex; flex-wrap:wrap; gap:10px;">')
        for f in files:
            rel = os.path.relpath(f, out_dir).replace("\\", "/")
            add(f'<div style="flex:0 0 auto;"><img src="{rel}" style="max-width:360px; border:1px solid #ddd; border-radius:8px;"><br><small>{os.path.basename(f)}</small></div>')
        add("</div>")

    add("<!DOCTYPE html><html><head><meta charset='utf-8'><title>XAI Report</title></head><body style='font-family:Segoe UI,Arial,sans-serif;'>")
    add("<h1>Multimodal AD — XAI Report</h1>")
    add("<p>Auto-generated: SHAP (global + per-row), PDP, LIME for tabular; Grad-CAM & mid-slices for MRI.</p>")

    img_grid("SHAP — Global", os.path.join(out_dir, "tabular_shap"))
    img_grid("SHAP — Dependence", os.path.join(out_dir, "tabular_shap", "dependence"))
    img_grid("SHAP — Per-row Waterfalls", os.path.join(out_dir, "tabular_shap", "waterfall"))
    img_grid("PDP (top features)", os.path.join(out_dir, "tabular_pdp"))
    img_grid("LIME (per-row)", os.path.join(out_dir, "tabular_lime"))
    img_grid("MRI (Grad-CAM overlays & mid-slice montages)", os.path.join(out_dir, "mri"))

    add("</body></html>")

    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"[OK] Built report: {out_html}")


# -----------------------
# MAIN
# -----------------------
def main():
    print("[1/5] Load & split tabular…")
    df = pd.read_csv(CSV)
    X_train, X_test, y_train, y_test, feat_names, scaler, dx_col, X_train_df, X_test_df = prepare_tabular(df)

    print("[2/5] Train tabular model for explanations…")
    rf = train_tabular_model(X_train, y_train)
    # quick sanity
    try:
        p = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        print(f"[INFO] RF AUROC on holdout: {auc:.3f}")
        print("[INFO] RF report:\n", classification_report(y_test, (p >= 0.5).astype(int)))
    except Exception:
        pass

    print("[3/5] Tabular SHAP…")
    shap_global_and_rowplots(rf, X_train, X_test, y_test, feat_names, TAB_SHAP_DIR)

    print("[4/5] Tabular PDP…")
    run_pdp(rf, X_train, feat_names, TAB_PDP_DIR, top_k=8)

    print("[5/5] Tabular LIME…")
    run_lime(rf, X_train, X_test, feat_names, TAB_LIME_DIR, max_rows=MAX_LIME)

    print("[MRI] Gather MRI figures (Grad-CAM + slices)…")
    gather_mri_figs(MRI_DIR, limit=TOPK)

    build_html_report(OUT)
    print(f"[DONE] XAI master finished -> {OUT}")


if __name__ == "__main__":
    main()
