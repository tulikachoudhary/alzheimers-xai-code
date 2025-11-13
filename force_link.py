# force_link.py  —  FIXED (.str.upper) and robust SITE+DATE linking for AD PTIDs
import os, re
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
RAW  = os.path.join(BASE, "ALL_NIFTI")
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic.csv")
ADNIMERGE = os.path.join(BASE, "ADNIMERGE.csv")

SITE_RE  = re.compile(r"^(\d{3})_")     # leading site code in filename
DATE_RE8 = re.compile(r"(\d{8})")       # YYYYMMDD in filename
PTID_RE  = re.compile(r"(\d{3}_S_\d{4})")

def canon_ptid(x:str)->str:
    s = str(x) if x is not None else ""
    m = PTID_RE.search(s)
    return m.group(1) if m else s.strip()

def to_date_any(x):
    d = pd.to_datetime(x, errors="coerce")
    return d.date() if pd.notna(d) else None

def to_date_ymd(s8):
    try:
        return datetime.strptime(s8, "%Y%m%d").date()
    except Exception:
        return None

def build_site_date_index(mri_dir):
    idx = {}
    nii_paths = [str(p) for p in Path(mri_dir).rglob("*.nii")] + [str(p) for p in Path(mri_dir).rglob("*.nii.gz")]
    for p in nii_paths:
        name = Path(p).name
        msite = SITE_RE.match(name)
        mdate = DATE_RE8.search(name)
        if not msite or not mdate:
            continue
        site = msite.group(1)
        dt = to_date_ymd(mdate.group(1))
        if not dt:
            continue
        idx.setdefault((site, dt), []).append(p)
    # prefer largest file for a given (site,date)
    for k, paths in idx.items():
        uniq = list(set(paths))
        uniq.sort(key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0, reverse=True)
        idx[k] = uniq
    return idx

def nearest_match(site, exam_dates, index, windows=(0,7,30,90,365,1095)):
    if not exam_dates: return None
    site_dates = sorted({dt for (s, dt) in index.keys() if s == site})
    if not site_dates: return None
    for W in windows:
        best_dt, best_gap = None, None
        for ed in exam_dates:
            for dt in site_dates:
                gap = abs((dt - ed).days)
                if gap <= W and (best_gap is None or gap < best_gap):
                    best_dt, best_gap = dt, gap
        if best_dt is not None:
            return index[(site, best_dt)][0]
    return None

def main():
    if not os.path.exists(CSV): raise SystemExit(f"[ERROR] Missing CSV: {CSV}")
    if not os.path.exists(ADNIMERGE): raise SystemExit(f"[ERROR] Missing ADNIMERGE: {ADNIMERGE}")

    df = pd.read_csv(CSV)
    if "PTID" not in df.columns:
        raise SystemExit("[ERROR] features_with_genetic.csv must include PTID.")
    if "PATH" not in df.columns:
        df["PATH"] = ""

    df["PTID_CANON"] = df["PTID"].astype(str).apply(canon_ptid)

    # ---- Load ADNIMERGE, find AD PTIDs + EXAMDATEs (FIX: use .str.upper()) ----
    am = pd.read_csv(ADNIMERGE, low_memory=False)
    am_ptid = next((c for c in am.columns if c.upper()=="PTID"), None)
    am_dx   = next((c for c in am.columns if c.upper().startswith("DX")), None)
    am_exam = next((c for c in am.columns if "EXAM" in c.upper() and "DATE" in c.upper()), None)
    if am_ptid is None or am_dx is None or am_exam is None:
        raise SystemExit("[ERROR] ADNIMERGE needs PTID, DX*, and EXAMDATE columns.")

    am["_PTID"] = am[am_ptid].astype(str).apply(canon_ptid)
    am["_DX"]   = am[am_dx].astype(str).str.upper().str.strip()   # <-- fixed here
    am["_EXAM"] = am[am_exam].apply(to_date_any)
    am = am.dropna(subset=["_EXAM"])

    AD_NAMES = {"AD","ALZHEIMER'S DISEASE","ALZHEIMER DISEASE","ALZHEIMERS DISEASE"}
    VALID = {"AD","MCI","CN","EMCI","LMCI"}

    dx_csv_col = next((c for c in df.columns if c.upper()=="DX_BL"), None) \
              or next((c for c in df.columns if str(c).upper().startswith("DX")), None)
    if dx_csv_col is None:
        raise SystemExit("[ERROR] No DX column (DX_BL/DX*) found in CSV.")
    df[dx_csv_col] = df[dx_csv_col].astype(str).str.upper().str.strip()
    df = df[df[dx_csv_col].isin(VALID)].copy()

    ptids_ad = set(am.loc[am["_DX"].isin(AD_NAMES), "_PTID"])
    ad_ptids_in_csv = sorted(set(df["PTID_CANON"]) & ptids_ad)
    if not ad_ptids_in_csv:
        print("[REPORT] No AD PTIDs from ADNIMERGE present in your CSV scope.")
        return

    # Build SITE+DATE index and PTID->EXAMDATEs
    site_date_index = build_site_date_index(RAW)
    ptid2dates = am.groupby("_PTID")["_EXAM"].apply(
        lambda s: sorted(list({d for d in s if pd.notna(d)}))
    ).to_dict()

    def ptid_site(ptid):
        m = re.match(r"^(\d{3})_S_", ptid)
        return m.group(1) if m else None

    updated_ptids = []
    for ptid in ad_ptids_in_csv:
        # skip if already linked
        has_any = df.loc[df["PTID_CANON"]==ptid, "PATH"].apply(lambda p: isinstance(p,str) and os.path.isfile(p)).any()
        if has_any:
            continue
        site = ptid_site(ptid)
        if not site:
            continue
        exam_dates = ptid2dates.get(ptid, [])
        hit = nearest_match(site, exam_dates, site_date_index)
        if hit:
            rows = (df["PTID_CANON"] == ptid)
            df.loc[rows, "PATH"] = os.path.abspath(hit)
            updated_ptids.append(ptid)

    df.to_csv(CSV, index=False)
    print(f"[REPORT] AD PTIDs present in CSV: {len(ad_ptids_in_csv)}")
    print(f"[REPORT] Newly linked AD PTIDs (PATH written): {len(updated_ptids)}")
    print("Sample newly linked:", updated_ptids[:10])

if __name__ == "__main__":
    main()
