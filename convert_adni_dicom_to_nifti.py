#!/usr/bin/env python3
r"""
ADNI DICOM -> NIfTI (robust, ADNI-structured folders)

Usage (Windows):
  python convert_adni_dicom_to_nifti.py -i "C:\Users\tulikachoudhary\Desktop\c502\ADNI" -o "C:\Users\tulikachoudhary\Desktop\c502\nifti"
"""

import argparse, os, sys, subprocess, time, shutil
from pathlib import Path
import re

# ---- Configure your local dcm2niix.exe (required) ----
DCM2NIIX = Path(r"C:\Users\tulikachoudhary\Desktop\c502\dcm2niix.exe")  # <- your working exe

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")  # e.g., 2022-07-08_09_21_14.0

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write(log: Path, msg: str, echo=True):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    if echo:
        print(msg, flush=True)

def run(cmd: str) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def dcm2niix_cmd(in_dir: Path, out_dir: Path) -> str:
    # -1 y: merge slices, -z y: gzip, -r y: recurse, -d 9: deep search, -v 3: verbose
    return f'"{DCM2NIIX}" -1 y -z y -r y -d 9 -v 3 -f "%p_%s" -o "{out_dir}" "{in_dir}"'

def looks_like_date_dir(p: Path) -> bool:
    return p.is_dir() and DATE_DIR_RE.match(p.name or "")

def dir_has_dicoms(p: Path) -> bool:
    try:
        return any(f.is_file() and f.suffix.lower()==".dcm" for f in p.iterdir())
    except Exception:
        return False

def find_adni_date_dirs(root: Path) -> list[Path]:
    """
    ADNI layout (common):
      .../<SUBJECT>/<SERIES_NAME>/<DATE_DIR>/<Ixxxxx>/file.dcm
    We want the DATE_DIR level (one above Ixxxxx).
    """
    date_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        # If this folder directly contains .dcm, mark its parent (DATE_DIR candidate)
        if any(fn.lower().endswith(".dcm") for fn in filenames):
            parent = p.parent
            # p is usually Ixxxxx; parent should be DATE_DIR
            if looks_like_date_dir(parent):
                date_dirs.add(parent)
            elif looks_like_date_dir(p):
                # sometimes .dcm are directly in DATE_DIR
                date_dirs.add(p)
    return sorted(date_dirs)

def try_convert(one_input: Path, out_dir: Path, log: Path) -> bool:
    """Run dcm2niix; return True if any NIfTI was written to out_dir."""
    ensure_dir(out_dir)
    cmd = dcm2niix_cmd(one_input, out_dir)
    write(log, f"CMD: {cmd}")
    code, out, err = run(cmd)
    if out: write(log, "stdout:\n" + out, echo=False)
    if err: write(log, "stderr:\n" + err, echo=False)
    made = list(out_dir.glob("*.nii*"))
    if code == 0 and made:
        return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Convert ADNI DICOM tree to NIfTI.")
    ap.add_argument("-i","--input", required=True, help="ADNI input root")
    ap.add_argument("-o","--output", required=True, help="Output root")
    args = ap.parse_args()

    in_root  = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    log = out_root / "convert_adni_to_nifti.log"

    write(log, f"=== START {time.ctime()} ===")
    write(log, f"INPUT : {in_root}")
    write(log, f"OUTPUT: {out_root}")

    if not DCM2NIIX.is_file():
        write(log, f"ERROR: dcm2niix.exe not found at {DCM2NIIX}")
        sys.exit(1)
    if not in_root.exists():
        write(log, "ERROR: input root does not exist.")
        sys.exit(1)

    ensure_dir(out_root)

    series = find_adni_date_dirs(in_root)
    write(log, f"Found {len(series)} DATE-level series folders.")

    ok = 0
    fail = 0
    for s in series:
        rel = s.relative_to(in_root)
        out_dir = out_root / rel
        write(log, f"\n=== Series: {s} ===")
        # 1) Try DATE folder (the level that worked in your manual test)
        if try_convert(s, out_dir, log):
            write(log, f"OK   : {s}")
            ok += 1
            continue

        # 2) If nothing written, try one level deeper for any Ixxxxx children
        deep_ok = False
        if s.exists():
            for child in s.iterdir():
                if child.is_dir() and child.name.upper().startswith("I") and dir_has_dicoms(child):
                    write(log, f"Retry deeper: {child}")
                    if try_convert(child, out_dir, log):
                        deep_ok = True
                        break
        if deep_ok:
            write(log, f"OK   : {s} (via child)")
            ok += 1
            continue

        # 3) Final retry: one level up (series folder)
        up = s.parent
        if up.exists():
            write(log, f"Retry up: {up}")
            if try_convert(up, out_dir, log):
                write(log, f"OK   : {s} (via parent)")
                ok += 1
                continue

        write(log, f"FAIL : {s} (no NIfTI written after retries)")
        fail += 1

    write(log, f"\nSUMMARY: ok={ok}, fail={fail}")
    write(log, f"=== END {time.ctime()} ===")

if __name__ == "__main__":
    main()
