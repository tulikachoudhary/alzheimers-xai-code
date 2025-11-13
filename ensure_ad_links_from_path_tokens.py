import os, re, glob, shutil
from pathlib import Path
import pandas as pd

BASE = r"C:\Users\TulikaChoudhary\Desktop\c502"
RAW  = os.path.join(BASE, "ALL_NIFTI")                 # source
DST  = os.path.join(BASE, "ALL_NIFTI_PROCESSED")       # destination
CSV  = os.path.join(BASE, "FEATURES", "features_with_genetic.csv")
ADNIMERGE = os.path.join(BASE, "ADNIMERGE.csv")

os.makedirs(DST, exist_ok=True)

PTID_RE = re.compile(r"(\d{3}_S_\d{4})")   # e.g., 016_S_6939

def pick(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

def rid_from_ptid(ptid: str):
    m = re.search(r"_S_(\d{4})$", ptid)
    return m.group(1) if m else None

def scan_raw_index():
    """Build maps:
       - ptid -> list of nii paths (from any part of the full path)
       - rid  -> list of nii paths (RID = last 4 digits of PTID, from path)
    """
    ptid_index = {}
    rid_index  = {}
    nii_paths = [str(p) for p in Path(RAW).rglob("*.nii")] + [str(p) for p in Path(RAW).rglob("*.nii.gz")]
    for p in nii_paths:
        full = p.replace("\\", "/")
        # 1) PTID in path
        m = PTID_RE.search(full)
        if m:
            pt = m.group(1)
            ptid_index.setdefault(pt, []).append(p)
            rid = rid_from_ptid(pt)
            if rid:
                rid_index.setdefault(rid, []).append(p)
        else:
            # 2) fallback: try any 4-digit RID-like token bounded by non-digits
            rid_hits = re.findall(r"(?<!\d)(\d{4})(?!\d)", Path(p).name)
            for rh in rid_hits:
                rid_index.setdefault(rh, []).append(p)
    # de-dup and prefer largest file first
    def sort_by_size(paths): 
        return sorted(set(paths), key=lambda x: os.path.getsize(x), reverse=True)
    ptid_index = {k: sort_by_size(v) for k, v in ptid_index.items()}
    rid_index  = {k: sort_by_size(v) for k, v in rid_index.items()}
    return ptid_index, rid_index

def main():
    if not os.path.exists(CSV): raise SystemExit(f"[ERROR] Missing CSV: {CSV}")
    if not os.path.exists(ADNIMERGE): raise SystemExit(f"[ERROR] Missing ADNIMERGE: {ADNIMERGE}")

    df = pd.read_csv(CSV)
    am = pd.read_csv(ADNIMERGE, low_memory=False)

    # Columns
    pt_c = pick(df, ["PTID","ptid"]); dx_c = pick(df, ["DX_bl","DX_BL","DX"])
    if pt_c is None or dx_c is None:
        raise SystemExit("[ERROR] CSV needs PTID and DX(DX_bl/DX_BL/DX).")
    if "PATH" not in df.columns:
        df["PATH"] = ""

    am_pt = pick(am, ["PTID","ptid"]); 
    if am_pt is None: 
        raise SystemExit("[ERROR] ADNIMERGE needs PTID column.")
    # AD set from ADNIMERGE (most reliable diagnosis source)
    am["_PTID"] = am[am_pt].astype(str).str.strip()
    # If ADNIMERGE has DX_bl, use that; else fallback to CSV’s DX
    am_dx = pick(am, ["DX_bl","DX_BL","DX"])
    if am_dx:
        am["_DX"] = am[am_dx].astype(str).str.upper().str.strip()
        ad_ptids = set(am.loc[am["_DX"].isin(["AD","ALZHEIMER'S DISEASE","ALZHEIMER DISEASE","ALZHEIMERS DISEASE"]), "_PTID"])
    else:
        # fallback: derive AD from features CSV
        ad_ptids = set(df.loc[df[dx_c].astype(str).str.upper().str.strip().isin(
            ["AD","ALZHEIMER'S DISEASE","ALZHEIMER DISEASE","ALZHEIMERS DISEASE"]
        ), pt_c].astype(str).str.strip())

    print(f"[INFO] AD PTIDs (source={'ADNIMERGE' if am_dx else 'CSV'}): {len(ad_ptids)}")

    # Build indexes from RAW
    print("[INFO] Indexing RAW by path tokens…")
    ptid_index, rid_index = scan_raw_index()
    print(f"[INFO] PTIDs seen in RAW paths: {len(ptid_index)} | RIDs seen: {len(rid_index)}")

    # Link logic: prefer exact PTID-in-path; else RID-in-name
    linked_ptids = set()
    updated_paths = 0

    for ptid in sorted(ad_ptids):
        # skip if already linked in CSV
        has_any = df.loc[df[pt_c].astype(str).str.strip()==ptid, "PATH"].apply(lambda p: isinstance(p, str) and os.path.isfile(p)).any()
        if has_any:
            linked_ptids.add(ptid)
            continue

        hits = ptid_index.get(ptid, [])
        if not hits:
            rid = rid_from_ptid(ptid)
            if rid and rid in rid_index:
                hits = rid_index[rid]

        if hits:
            best = hits[0]  # largest by size
            # write PATH for all rows of this PTID (only AD rows will be used by trainer label=1)
            rows = df[pt_c].astype(str).str.strip() == ptid
            df.loc[rows, "PATH"] = os.path.abspath(best)
            linked_ptids.add(ptid)
            updated_paths += int(rows.sum())

    # Write CSV back
    out_csv = CSV  # in-place update
    df.to_csv(out_csv, index=False)
    print(f"[REPORT] Updated PATH cells: {updated_paths}")
    print(f"[REPORT] Distinct AD PTIDs with resolvable MRI now: {len(linked_ptids)}")
    print("Sample linked PTIDs:", list(sorted(linked_ptids))[:10])

    # Optionally copy 1 file per linked PTID into PROCESSED so other tools see them
    copied = 0
    seen_dst = {Path(p).name for p in glob.glob(os.path.join(DST, "*.nii*"))}
    for ptid in linked_ptids:
        cand = ptid_index.get(ptid)
        if not cand:
            rid = rid_from_ptid(ptid)
            if rid: cand = rid_index.get(rid, [])
        if not cand: continue
        src = cand[0]
        bn = Path(src).name
        if bn not in seen_dst:
            shutil.copy2(src, os.path.join(DST, bn))
            seen_dst.add(bn)
            copied += 1
    print(f"[REPORT] Copied {copied} NIfTI files into PROCESSED.")

if __name__ == "__main__":
    main()
