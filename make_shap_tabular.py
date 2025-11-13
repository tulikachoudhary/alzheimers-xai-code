# make_shap_tabular.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic_rowwise.csv")
OUT  = os.path.join(BASE, "VIS", "shap_tabular")
os.makedirs(OUT, exist_ok=True)

RNG = 42
N_BG = 100           # background size for KernelExplainer
N_SAMPLE = 300       # rows to explain (kept modest for speed)
TOPK = 10            # top-k features for bar + dependence

def build_feature_matrix(df):
    # numeric only; drop obvious leakage/IDs if they exist
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    drop_cols = [c for c in X.columns if any(k in c.upper() for k in ban)]
    X = X.drop(columns=drop_cols, errors="ignore").fillna(0).astype(np.float32)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    return pd.DataFrame(Xs, columns=X.columns, index=df.index), sc

def main():
    df = pd.read_csv(CSV)
    # label (AD vs Non-AD)
    dx_col = next((c for c in df.columns if c.upper()=="DX_BL"), None) \
          or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx_col is None:
        raise SystemExit("[FATAL] Need DX or DX_BL column.")
    keep = df[dx_col].astype(str).str.upper().isin({"AD","MCI","CN"})
    df = df.loc[keep].copy()
    y = (df[dx_col].astype(str).str.upper()=="AD").astype(int).values

    X, scaler = build_feature_matrix(df)

    # small, fast surrogate to interpret tabular side
    X_tr, X_va, y_tr, y_va = train_test_split(
        X.values, y, test_size=0.2, random_state=RNG, stratify=y if len(np.unique(y))>1 else None
    )
    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RNG)
    clf.fit(X_tr, y_tr)
    print("[REPORT] Tabular surrogate (LR):")
    print(classification_report(y_va, clf.predict(X_va), digits=3))

    # SHAP with KernelExplainer (works with any sklearn model)
    # Background: random subset
    rng = np.random.default_rng(RNG)
    bg_idx = rng.choice(X.shape[0], size=min(N_BG, X.shape[0]), replace=False)
    bg = X.iloc[bg_idx]

    # Sample to explain
    smp_idx = rng.choice(X.shape[0], size=min(N_SAMPLE, X.shape[0]), replace=False)
    sample = X.iloc[smp_idx]

    f = lambda data: clf.predict_proba(data)[:,1]  # predict AD probability
    expl = shap.KernelExplainer(f, bg, link="logit")
    shap_values = expl.shap_values(sample, nsamples="auto")  # returns (n_samples, n_features)

    # ----- Plots -----
    # Beeswarm
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, sample, feature_names=sample.columns, show=False)
    plt.tight_layout()
    bees = os.path.join(OUT, "shap_beeswarm.png"); plt.savefig(bees, dpi=150); plt.close()
    print("[OK] wrote", bees)

    # Mean |SHAP| bar (top-k)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOPK]
    top_names = sample.columns[order]
    top_vals  = mean_abs[order]
    plt.figure(figsize=(8,5))
    plt.barh(range(len(top_names))[::-1], top_vals[::-1])
    plt.yticks(range(len(top_names))[::-1], top_names[::-1], fontsize=8)
    plt.xlabel("Mean |SHAP|"); plt.title(f"Top {len(top_names)} features")
    plt.tight_layout()
    barp = os.path.join(OUT, "shap_bar_topk.png"); plt.savefig(barp, dpi=150); plt.close()
    print("[OK] wrote", barp)

    # Dependence plots for top-k
    for i, fname in enumerate(top_names):
        plt.figure(figsize=(6,5))
        shap.dependence_plot(
            fname, shap_values, sample, feature_names=sample.columns, show=False, interaction_index=None
        )
        plt.tight_layout()
        dep = os.path.join(OUT, f"shap_dependence_{i+1:02d}_{fname.replace('/','-')}.png")
        plt.savefig(dep, dpi=150); plt.close()
        print("[OK] wrote", dep)

    print("[DONE] SHAP tabular →", OUT)

if __name__ == "__main__":
    main()
