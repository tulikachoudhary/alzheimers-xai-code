import pandas as pd
from pathlib import Path

# ===== paths =====
ROOT = Path(r"C:\Users\tulikachoudhary\Desktop\c502")
OUT  = ROOT / "clinical_baseline.csv"

# required files (after renaming)
FILES = {
    "ADNIMERGE": ROOT / "ADNIMERGE.csv",
    "MMSE":      ROOT / "MMSE.csv",
    "CDR":       ROOT / "CDR.csv",
    "ADAS":      ROOT / "ADAS.csv",
    "PTDEMOG":   ROOT / "PTDEMOG.csv",
    "FAQ":       ROOT / "FAQ.csv",
    "RAVLT":     ROOT / "RAVLT.csv",  # your Neurobat-derived file
}

def read_csv_safe(path):
    return pd.read_csv(path, low_memory=False)

# 1) core table & baseline filter
adni = read_csv_safe(FILES["ADNIMERGE"])
adni["VISCODE"] = adni["VISCODE"].astype(str).str.lower()
bl = adni[adni["VISCODE"].eq("bl")].copy()

# choose diagnosis column
dx_col = "DX_bl" if "DX_bl" in bl.columns else ("DX" if "DX" in bl.columns else None)
if dx_col is None:
    raise ValueError("DX/DX_bl not found in ADNIMERGE.csv")

keep_cols = [c for c in ["RID","PTID","VISCODE","EXAMDATE","AGE","PTGENDER","PTEDUCAT", dx_col] if c in bl.columns]
clinical = bl[keep_cols].rename(columns={dx_col: "DX_BASELINE"})

def merge_visit_table(main_df, csv_path, prefix):
    if not csv_path.exists():
        return main_df
    t = read_csv_safe(csv_path)

    # normalize visit column name
    if "VISCODE" not in t.columns:
        for alt in ("VISCODE2","VISIT","VISCODE2_x","VISITCODE"):
            if alt in t.columns:
                t = t.rename(columns={alt: "VISCODE"})
                break
    if "VISCODE" in t.columns:
        t["VISCODE"] = t["VISCODE"].astype(str).str.lower()
        t = t[t["VISCODE"].eq("bl")]

    # keep essential keys + non-duplicate score columns
    base_keep = [c for c in ("RID","VISCODE","EXAMDATE") if c in t.columns]
    score_cols = [c for c in t.columns if c not in base_keep]
    # be conservative: drop obviously duplicate identifiers
    drop_like = {"PTID","SITE","PHASE","COLPROT","PROJECT","VISNAME"}
    score_cols = [c for c in score_cols if c not in drop_like]
    t = t[base_keep + score_cols]

    # prefix score columns to avoid name clashes
    rename = {c: f"{prefix}{c}" for c in score_cols}
    t = t.rename(columns=rename)

    return main_df.merge(t, on=[c for c in ("RID","VISCODE") if c in main_df.columns and c in t.columns], how="left")

# 2) merge MMSE, CDR, ADAS, FAQ, RAVLT
clinical = merge_visit_table(clinical, FILES["MMSE"],  "MMSE_")
clinical = merge_visit_table(clinical, FILES["CDR"],   "CDR_")    # expect CDR_GLOBAL/CDRSB etc.
clinical = merge_visit_table(clinical, FILES["ADAS"],  "ADAS_")   # expect ADAS11/ADAS13 etc.
clinical = merge_visit_table(clinical, FILES["FAQ"],   "FAQ_")    # expect FAQTOTAL
clinical = merge_visit_table(clinical, FILES["RAVLT"], "RAVLT_")  # expect RAVLT_* measures

# 3) add demographics if missing
if FILES["PTDEMOG"].exists():
    demo = read_csv_safe(FILES["PTDEMOG"]).copy()
    demo = demo[[c for c in demo.columns if c in {"RID","PTGENDER","PTEDUCAT","PTMARRY","PTRACCAT","PTETHCAT"}]].drop_duplicates("RID")
    for c in demo.columns:
        if c != "RID" and c not in clinical.columns:
            clinical = clinical.merge(demo[["RID", c]], on="RID", how="left")

# 4) light cleanup
if clinical["DX_BASELINE"].dtype == object:
    clinical["DX_BASELINE"] = clinical["DX_BASELINE"].str.strip()

# 5) save
clinical.to_csv(OUT, index=False)

# 6) tiny report
print(f"Saved: {OUT}")
if "DX_BASELINE" in clinical.columns:
    print("\nCounts by DX_BASELINE:")
    print(clinical["DX_BASELINE"].value_counts(dropna=False))
print("\nColumns:", list(clinical.columns))
print("\nPreview:")
print(clinical.head(8))
