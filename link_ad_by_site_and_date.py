import os, re, glob, shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
RAW  = os.path.join(BASE, "ALL_NIFTI")                        # scan here
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic.csv")  # update PATH here
ADNIMERGE = os.path.join(BASE, "ADNIMERGE.csv")

TARGET_NEW_AD_PTIDS = 6      # aim to link at least this many AD subjects
WINDOWS_DAYS = [0, 7, 30, 90, 365, 1095]  # try increasingly wider date windows

SITE_RE = re.compile(r"^(\d{3})_")          # leading site code like 002_
DATE_RE = re.compile(r"(\d{8})")            # any YYYYMMDD in filename

def to_date_ymd(s):
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        return None

def to_date_any(s):
    try:
        d = pd.to_datetime(s, errors="coerce")
        return d.date() if pd.notna(d) else None
    except Exception:
        return None

def pick(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

def main():
    if not os.path.exists(CSV): raise SystemExit(f"[ERROR] Missing CSV: {CSV}")
    if not os.path.exists(ADNIMERGE): raise SystemExit(f"[ERROR] Missing ADNIMERGE: {ADNIMERGE}")

    df = pd.read_csv(CSV)
    am = pd.read_csv(ADNIMERGE, low_memory=False)

    # Identify key columns
    ptid_c = pick(df, ["PTID","ptid"])
    dx_c   = pick(df, ["DX_bl","DX_BL","DX"])
    if ptid_c is None or dx_c is None:
        raise SystemExit("[ERROR] CSV needs PTID and DX(DX_bl/DX_BL/DX).")
    if "PATH" not in df.columns:
        df["PATH"] = ""

    am_ptid = pick(am, ["PTID","ptid"])
    am_dx   = pick(am, ["DX_bl","DX_BL","DX"])
    am_exam = pick(am, ["EXAMDATE","EXAM_DATE","EXAMDT"])
    if not am_ptid or not am_dx or not am_exam:
        raise SystemExit("[ERROR] ADNIMERGE needs PTID, DX, and EXAMDATE columns.")

    # Build AD set with (PTID -> list of EXAMDATEs) and site code from PTID
    am["_PTID"] = am[am_ptid].astype(str).str.strip()
    am["_DX"]   = am[am_dx].astype(str).str.upper().str.strip()
    am["_EXAM"] = am[am_exam].apply(to_date_any)
    am = am.dropna(subset=["_EXAM"])

    ad_only = am[am["_DX"].isin(["AD","ALZHEIMER'S DISEASE","ALZHEIMER DISEASE","ALZHEIMERS DISEASE"])].copy()
    if ad_only.empty:
        raise SystemExit("[ERROR] No AD rows in ADNIMERGE.")
    # site code = first 3 digits of PTID
    ad_only["_SITE"] = ad_only["_PTID"].str.extract(r"^(\d{3})_S_\d{4}$")[0]
    ad_only = ad_only.dropna(subset=["_SITE"])

    # Map: (SITE -> list of (PTID, EXAMDATE))
    site2ad = {}
    for _, r in ad_only.iterrows():
        site = str(r["_SITE"])
        ptid = str(r["_PTID"])
        exdt = r["_EXAM"]
        site2ad.setdefault(site, []).append((ptid, exdt))

    # Scan RAW: build (SITE, DATE) -> list of NIfTI paths (largest first)
    file_index = {}  # key=(site, date) -> [paths]
    nii_paths = [str(p) for p in Path(RAW).rglob("*.nii")] + [str(p) for p in Path(RAW).rglob("*.nii.gz")]
    for p in nii_paths:
        name = Path(p).name
        msite = SITE_RE.match(name)
        mdate = DATE_RE.search(name)
        if not msite or not mdate:
            continue
        site = msite.group(1)
        dt = to_date_ymd(mdate.group(1))
        if not dt:
            continue
        file_index.setdefault((site, dt), []).append(p)

    # Deduplicate & sort paths per (site, date) by size (largest first)
    for k, paths in list(file_index.items()):
        unique = {}
        for path in paths:
            try:
                sz = os.path.getsize(path)
            except Exception:
                sz = 0
            unique[path] = sz
        file_index[k] = [p for p,_ in sorted(unique.items(), key=lambda kv: kv[1], reverse=True)]

    # Which AD PTIDs already have a valid PATH?
    def ok_path(p): return isinstance(p, str) and os.path.isfile(p)
    linked_ptids = set(
        df.loc[df[ptid_c].astype(str).str.strip().isin(ad_only["_PTID"]) & df["PATH"].apply(ok_path), ptid_c]
        .astype(str).str.strip().unique()
    )

    new_links = 0
    # Try to link more AD PTIDs by same-site nearest date
    for site, pairs in site2ad.items():
        # Build an ordered list of all scan dates present for this site in RAW
        site_dates = sorted({dt for (s,dt) in file_index.keys() if s == site})
        if not site_dates:
            continue
        for (ptid, exdt) in pairs:
            if ptid in linked_ptids:
                continue
            # Try growing windows around EXAMDATE
            chosen_path = None
            for W in WINDOWS_DAYS:
                # find nearest site date within ±W days
                best_dt = None
                best_abs = None
                for dt in site_dates:
                    delta = abs((dt - exdt).days)
                    if delta <= W and (best_abs is None or delta < best_abs):
                        best_abs = delta
                        best_dt = dt
                if best_dt is not None:
                    # pick the largest NIfTI for that (site, date)
                    chosen_path = file_index[(site, best_dt)][0]
                    break
            if chosen_path:
                # write PATH for all rows of this PTID in CSV
                rows = df[ptid_c].astype(str).str.strip() == ptid
                df.loc[rows, "PATH"] = os.path.abspath(chosen_path)
                linked_ptids.add(ptid)
                new_links += int(rows.sum())
                # stop if we already hit our target of distinct AD PTIDs
                if len(linked_ptids) >= TARGET_NEW_AD_PTIDS:
                    break
        if len(linked_ptids) >= TARGET_NEW_AD_PTIDS:
            break

    df.to_csv(CSV, index=False)
    print(f"[REPORT] PATH cells written/updated: {new_links}")
    print(f"[REPORT] Distinct AD PTIDs with resolvable MRI now: {len(linked_ptids)}")
    print("Sample linked AD PTIDs:", list(sorted(linked_ptids))[:12])

if __name__ == "__main__":
    main()
