import os, glob, shutil, ntpath
import pandas as pd

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic.csv")   # change if needed
RAW  = os.path.join(BASE, "ALL_NIFTI")                               # source
DST  = os.path.join(BASE, "ALL_NIFTI_PROCESSED")                      # destination
TARGET_AD_PTIDS = 6  # ensure at least this many AD subjects are resolvable in PROCESSED

os.makedirs(DST, exist_ok=True)

def rid_or_none(x):
    try:
        if pd.isna(x): return None
        return int(float(x))
    except Exception:
        return None

def pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- Load CSV robustly ---
df = pd.read_csv(CSV)

# Diagnosis column (DX_bl vs DX_BL vs DX)
dx_col = pick_first_existing_col(df, ["DX_bl","DX_BL","DX"])
if not dx_col:
    raise RuntimeError("No diagnosis column found. Expected one of: DX_bl, DX_BL, DX")

dx_norm = df[dx_col].astype(str).str.upper().str.strip()
ad = df[dx_norm.isin(["AD","ALZHEIMER'S DISEASE","ALZHEIMERS DISEASE","ALZHEIMER DISEASE"])].copy()

# Key columns
ptid_col = pick_first_existing_col(df, ["PTID","ptid"])
rid_col  = pick_first_existing_col(df, ["RID","rid"])
subj_col = pick_first_existing_col(df, ["SUBJECT","subject","subject_stem","SUBJECT_STEM"])

if not any([ptid_col, rid_col, subj_col]):
    raise RuntimeError("No identifier columns found. Expected one of PTID/RID/SUBJECT (any case).")

def row_tokens(r):
    ptid = str(r.get(ptid_col,"")).strip() if ptid_col else ""
    rid  = rid_or_none(r.get(rid_col,None)) if rid_col else None
    subj = str(r.get(subj_col,"")).strip() if subj_col else ""
    return ptid, rid, subj

rows = [row_tokens(r) for _, r in ad.iterrows()]

# Unique by PTID (fallback to RID/SUBJECT)
seen = set()
unique = []
for ptid, rid, subj in rows:
    key = ptid or (f"RID:{rid}" if rid is not None else "") or subj
    if key and key not in seen:
        seen.add(key)
        unique.append((ptid, rid, subj))

def find_hits(token):
    if not token: return []
    pats = [f"**/*{token}*.nii", f"**/*{token}*.nii.gz"]
    hits = []
    for p in pats:
        hits += glob.glob(os.path.join(RAW, p), recursive=True)
    return hits

def copy_first_hit(hits, dst_dir, seen_names):
    for src in hits:
        bn = ntpath.basename(src)
        if bn in seen_names: 
            continue
        shutil.copy2(src, os.path.join(dst_dir, bn))
        seen_names.add(bn)
        return bn
    return None

# What’s already resolvable in DST?
def resolvable_in_dst(ptid, rid, subj):
    checks = []
    if ptid: checks.append(glob.glob(os.path.join(DST, f"*{ptid}*.nii*")))
    if rid is not None:
        rid_str = str(rid)
        checks.append(glob.glob(os.path.join(DST, f"*{rid_str}*.nii*")))
        if rid_str.isdigit():
            checks.append(glob.glob(os.path.join(DST, f"*{rid_str.zfill(4)}*.nii*")))
    if subj: checks.append(glob.glob(os.path.join(DST, f"*{subj}*.nii*")))
    return any(len(c)>0 for c in checks)

resolvable_ptids = set()
for ptid, rid, subj in unique:
    if resolvable_in_dst(ptid, rid, subj):
        resolvable_ptids.add(ptid or (f"RID:{rid}" if rid is not None else subj))

seen_dst = set(ntpath.basename(p) for p in glob.glob(os.path.join(DST, "*.nii*")))
copied = 0

for ptid, rid, subj in unique:
    if len(resolvable_ptids) >= TARGET_AD_PTIDS:
        break
    if ptid and ptid in resolvable_ptids:
        continue

    bn = None
    # 1) PTID
    if ptid:
        bn = copy_first_hit(find_hits(ptid), DST, seen_dst)
    # 2) RID (plus zero-padded)
    if bn is None and rid is not None:
        rid_str = str(rid)
        bn = copy_first_hit(find_hits(rid_str), DST, seen_dst)
        if bn is None and rid_str.isdigit():
            bn = copy_first_hit(find_hits(rid_str.zfill(4)), DST, seen_dst)
    # 3) SUBJECT
    if bn is None and subj:
        bn = copy_first_hit(find_hits(subj), DST, seen_dst)

    if bn is not None:
        resolvable_ptids.add(ptid or (f"RID:{rid}" if rid is not None else subj))
        copied += 1

print(f"[REPORT] AD rows in CSV: {len(ad)}")
print(f"[REPORT] Unique AD IDs (by PTID/RID/SUBJECT): {len(unique)}")
print(f"[REPORT] Already resolvable in PROCESSED before copy: {len(resolvable_ptids)}")
print(f"[DONE] Copied {copied} NIfTI files into PROCESSED.")
print(f"[REPORT] AD IDs resolvable in PROCESSED now: {len(resolvable_ptids)}")
print("Sample resolvable AD IDs:", list(resolvable_ptids)[:10])
