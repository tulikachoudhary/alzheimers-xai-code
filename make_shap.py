import os, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

BASE=r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV =os.path.join(BASE,"FEATURES","features_with_genetic_rowwise.csv")
OUT =os.path.join(BASE,"VIS","XAI","SHAP"); os.makedirs(OUT, exist_ok=True)

def build_features(df):
    ban = ["DX","DIAG","PATH","PTID","RID","SUBJECT","SCANDATE","EXAMDATE","VISCODE"]
    X = df.select_dtypes(include=["number","bool"]).copy()
    X = X.drop(columns=[c for c in X.columns if any(k in c.upper() for k in ban)], errors="ignore")
    X = X.fillna(0).astype(np.float32)
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    return pd.DataFrame(Xs, columns=X.columns, index=df.index), sc

def main():
    df = pd.read_csv(CSV)
    dx = next((c for c in df.columns if c.upper()=="DX_BL"), None) or \
         next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    df = df[df[dx].astype(str).str.upper().isin(["AD","MCI","CN"])]
    y = (df[dx].astype(str).str.upper()=="AD").astype(int).values
    X, _ = build_features(df)

    # small train/test for surrogate tabular model
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # SHAP (TreeExplainer is fast for forests)
    explainer = shap.TreeExplainer(clf)
    # use up to 1000 samples to keep it quick
    sample = X.sample(min(1000, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # 1) Global summary (bar + beeswarm)
    plt.figure()
    shap.summary_plot(shap_values[1], sample, feature_names=sample.columns, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"shap_summary_beeswarm.png"), dpi=180); plt.close()

    plt.figure()
    shap.summary_plot(shap_values[1], sample, feature_names=sample.columns, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"shap_summary_bar.png"), dpi=180); plt.close()

    # 2) Top features: dependence plots (auto-pick top 6)
    # mean |SHAP| ranking
    mean_abs = np.abs(shap_values[1]).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:6]
    top_feats = [sample.columns[i] for i in top_idx]
    for f in top_feats:
        plt.figure()
        shap.dependence_plot(f, shap_values[1], sample, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"depend_{f}.png"), dpi=180); plt.close()

    # 3) Per-row force plot for 5 examples
    shap.initjs()  # enables color scale
    ex_rows = sample.iloc[:5]
    sv = explainer.shap_values(ex_rows)
    for i,(idx,row) in enumerate(ex_rows.iterrows()):
        fig = shap.force_plot(explainer.expected_value[1], sv[1][i,:], row, matplotlib=True, show=False)
        fig.savefig(os.path.join(OUT, f"force_row_{i}.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("[DONE] SHAP →", OUT)
if __name__ == "__main__":
    main()
