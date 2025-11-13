# merge_genetic_with_adni.py
# Merge MRI+clinical features with APOE genotypes.
# Robust to duplicate RID/PTID columns and messy APOE formats.

import re
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\tulikachoudhary\Desktop\c502")
FEATURES_PATH = BASE / "FEATURES" / "features_with_adnimerge.csv"
APOE_PATH     = BASE / "GENETIC"  / "ADNI_APOE_Genotype.csv"
OUT_PATH      = BASE / "FEATURES" / "features_with_genetic.csv"

def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.upper()
    )
    return df

def normalize_ptid_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.replace("-", "_", regex=False)
    def fix(x):
        m = re.match(r"^(\d{1,3})_S_(\d{1,4})$", x)
        if m:
            return f"{m.group(1).zfill(3)}_S_{m.group(2).zfill(4)}"
        return x
    return s.map(fix)

def coalesce_exact_name(df: pd.DataFrame, name: str) -> pd.Series | None:
    """
    Return a single Series by coalescing all columns whose name (case-insensitive)
    is exactly 'name'. Works even when there are true duplicate labels.
    """
    mask = df.columns.str.upper() == name.upper()
    if not mask.any():
        return None
    block = df.loc[:, mask]               # all RID (or PTID) duplicates as a DataFrame
    # coalesce left-to-right (first non-null per row)
    return block.bfill(axis=1).iloc[:, 0]

def parse_apoe(apoe_raw: pd.DataFrame) -> pd.DataFrame:
    apoe = normalize_columns(apoe_raw.copy())

    # IDs
    if "PTID" not in apoe.columns:
        for c in ["SUBJECT", "SUBJECT_ID", "ADNIID"]:
            if c in apoe.columns:
                apoe.rename(columns={c: "PTID"}, inplace=True)
                break
    if "PTID" in apoe.columns:
        apoe["PTID"] = normalize_ptid_series(apoe["PTID"])
    if "RID" in apoe.columns:
        apoe["RID"] = pd.to_numeric(apoe["RID"], errors="coerce").astype("Int64")

    # genotype source columns
    geno = next((c for c in apoe.columns if c in {"APOE","APOE_GENOTYPE","APOE ALLELES","GENOTYPE"}), None)
    a1 = next((c for c in apoe.columns if c in {"ALLELE1","APOE_ALLELE1","APOE1"}), None)
    a2 = next((c for c in apoe.columns if c in {"ALLELE2","APOE_ALLELE2","APOE2"}), None)
    num4 = next((c for c in apoe.columns if c in {"APOE4","APOE_4","NUME4","E4","APOE_E4_COUNT"}), None)

    if geno is None:
        if a1 and a2:
            apoe["APOE_GENOTYPE"] = (
                apoe[a1].astype(str).str.upper().str.replace("ε","E")
                + "/" +
                apoe[a2].astype(str).str.upper().str.replace("ε","E")
            )
        elif num4:
            apoe["APOE_GENOTYPE"] = apoe[num4].map({0:"E3/E3",1:"E3/E4",2:"E4/E4"}).fillna("UNK")
        else:
            apoe["APOE_GENOTYPE"] = "UNK"
    else:
        apoe["APOE_GENOTYPE"] = apoe[geno].astype(str).str.upper().str.replace("ε","E")

    # normalize formats like 34, 3/4, E3E4, e3/e4 → E3/E4
    def _norm_apoe(g):
        s = str(g).strip().upper().replace("ε","E").replace(" ", "")
        m = re.match(r"^E?([234])/?E?([234])$", s)   # 34, 3/4, E3/4, 3/E4
        if m:
            return f"E{m.group(1)}/E{m.group(2)}"
        m = re.match(r"^E([234])E([234])$", s)       # E3E4
        if m:
            return f"E{m.group(1)}/E{m.group(2)}"
        return s

    apoe["APOE_GENOTYPE"] = apoe["APOE_GENOTYPE"].map(_norm_apoe)
    apoe["APOE4_COUNT"]   = apoe["APOE_GENOTYPE"].str.count("E4")
    apoe["APOE4_CARRIER"] = apoe["APOE4_COUNT"] > 0

    keep = [c for c in ["PTID","RID","APOE_GENOTYPE","APOE4_COUNT","APOE4_CARRIER"] if c in apoe.columns]
    return apoe[keep].drop_duplicates()

def main():
    print("[1/6] Loading features (MRI + clinical)...")
    features = pd.read_csv(FEATURES_PATH)
    features = normalize_columns(features)

    # ----- PTID: coalesce duplicates first, then normalize text
    ptid_series = coalesce_exact_name(features, "PTID")
    if ptid_series is not None:
        features["PTID"] = normalize_ptid_series(ptid_series)

        # drop all but one PTID column (keep the canonical one we just set)
        mask = features.columns.str.upper() == "PTID"
        if mask.sum() > 1:
            # keep the last column (the one we assigned)
            to_drop = features.columns[mask][:-1]
            features.drop(columns=to_drop, inplace=True, errors="ignore")

    # ----- RID: coalesce duplicates to a single Series; only then convert numeric
    rid_series = coalesce_exact_name(features, "RID")
    if rid_series is not None:
        features["RID"] = pd.to_numeric(rid_series, errors="coerce").astype("Int64")
        # drop all but one RID column
        mask = features.columns.str.upper() == "RID"
        if mask.sum() > 1:
            to_drop = features.columns[mask][:-1]
            features.drop(columns=to_drop, inplace=True, errors="ignore")

    print("[2/6] Loading APOE genotype file...")
    apoe_raw = pd.read_csv(APOE_PATH)
    apoe = parse_apoe(apoe_raw)

    # choose merge key (prefer PTID)
    key = "PTID" if ("PTID" in apoe.columns and "PTID" in features.columns) else (
          "RID"  if ("RID"  in apoe.columns and "RID"  in features.columns)  else None)
    if key is None:
        print("[ERROR] No common ID column (PTID or RID) found in both files.")
        print("Feature cols:", list(features.columns)[:40])
        print("APOE cols:", list(apoe.columns))
        return

    print(f"[3/6] Merging on {key} …")
    merged = features.merge(apoe, on=key, how="left", validate="m:1")

    print("[4/6] Writing output …")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    total = len(merged)
    apoe_rows = merged["APOE_GENOTYPE"].notna().sum() if "APOE_GENOTYPE" in merged.columns else 0
    carriers  = merged["APOE4_CARRIER"].sum() if "APOE4_CARRIER" in merged.columns else 0

    print(f"[REPORT] Rows: {total}")
    print(f"[REPORT] With APOE genotype: {apoe_rows}")
    print(f"[REPORT] APOE4 carriers: {carriers}")
    print(f"[DONE] Saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
