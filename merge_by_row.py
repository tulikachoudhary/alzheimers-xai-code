import os, pandas as pd, sys

LEFT  = r".\FEATURES\features_with_adnimerge.csv"
RIGHT = r".\GENETIC\ADNI_APOE_Genotype.csv"
OUT   = r".\FEATURES\features_with_genetic_rowwise.csv"
MODE  = "clip"  # "strict", "clip", or "pad"

def load_csv(p):
    if not os.path.exists(p):
        sys.exit(f"[FATAL] File not found: {p}")
    df = pd.read_csv(p)
    drop_ix = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if drop_ix:
        df.drop(columns=drop_ix, inplace=True, errors="ignore")
    return df

L = load_csv(LEFT)
R = load_csv(RIGHT)

print(f"[INFO] LEFT  rows={len(L)}, cols={L.shape[1]}")
print(f"[INFO] RIGHT rows={len(R)}, cols={R.shape[1]}")

# Prefix genetic columns to avoid duplicates
pref = "GEN_"
R.columns = [(c if c not in L.columns and c != "" else f"{pref}{c or 'col'}") for c in R.columns]

# Handle unequal lengths
if len(L) != len(R):
    print(f"[WARN] Row counts differ (LEFT={len(L)}, RIGHT={len(R)}); MODE={MODE}")
    if MODE == "clip":
        n = min(len(L), len(R))
        L, R = L.iloc[:n].reset_index(drop=True), R.iloc[:n].reset_index(drop=True)
        print(f"[INFO] Clipped to {n} rows")
    elif MODE == "pad":
        if len(L) < len(R):
            pad = pd.DataFrame(index=range(len(R)-len(L)), columns=L.columns)
            L = pd.concat([L, pad], ignore_index=True)
        else:
            pad = pd.DataFrame(index=range(len(L)-len(R)), columns=R.columns)
            R = pd.concat([R, pad], ignore_index=True)

L, R = L.reset_index(drop=True), R.reset_index(drop=True)
merged = pd.concat([L, R], axis=1)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
merged.to_csv(OUT, index=False)
print(f"[DONE] Wrote -> {OUT}")
print(f"[REPORT] Final shape: rows={len(merged)}, cols={merged.shape[1]}")
