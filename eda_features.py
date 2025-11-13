import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\tulikachoudhary\Desktop\c502\FEATURES"
csv  = os.path.join(BASE, "features.csv")
outd = os.path.join(BASE, "eda")
os.makedirs(outd, exist_ok=True)

df = pd.read_csv(csv)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== SUMMARY ===")
# Compatible with all pandas versions
summary = df.select_dtypes(include=[np.number]).describe().T
print(summary)
summary.to_csv(os.path.join(outd, "summary.csv"))

# Histograms for main volume/intensity features
for col in ["ICV_ml", "GM_ml", "WM_ml", "CSF_ml", "int_mean", "int_std"]:
    if col in df.columns:
        plt.figure()
        df[col].dropna().plot(kind="hist", bins=15, edgecolor="black", color="steelblue")
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outd, f"{col}_hist.png"))
        plt.close()

# GM vs WM scatter
if {"GM_ml", "WM_ml"}.issubset(df.columns):
    plt.figure()
    plt.scatter(df["GM_ml"], df["WM_ml"], color="purple")
    plt.xlabel("GM_ml (ml)")
    plt.ylabel("WM_ml (ml)")
    plt.title("GM vs WM Volumes")
    plt.tight_layout()
    plt.savefig(os.path.join(outd, "gm_vs_wm_scatter.png"))
    plt.close()

print(f"\nEDA complete ✅ — results saved in: {outd}")
