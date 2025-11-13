import subprocess
from pathlib import Path
import sys

# -------- CONFIG (match your project) --------
C502 = Path(r"C:\Users\tulikachoudhary\Desktop\c502")
ADNI_ROOT = C502 / "ADNI"
ALL_NIFTI = C502 / "ALL_NIFTI"
PREPROC   = C502 / "PREPROC"
FEATURES  = C502 / "FEATURES"
MODELS    = C502 / "MODELS"
ADNIMERGE = C502 / "ADNIMERGE.csv"

# These scripts should already exist (you used them earlier)
SCRIPTS = {
    "convert":      "convert_dicom_batch.py",     # provided above
    "preprocess":   "preprocess_mri.py",
    "features":     "extract_features.py",
    "merge":        "merge_features_with_adni.py",
    "train_multi":  "train_baseline.py",          # AD vs LMCI vs CN (optional)
    "train_ad":     "train_ad_baseline.py",       # AD vs non-AD
    "score_ad":     "score_all_ad.py",
}
# ---------------------------------------------

def run_py(script, *args):
    path = C502 / script
    if not path.exists():
        print(f"[FATAL] Missing script: {path}")
        sys.exit(1)
    cmd = ["python", str(path), *args]
    print(f"\n=== RUN: {script} {' '.join(args)} ===")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[FATAL] {script} failed (code {res.returncode}). Aborting.")
        sys.exit(res.returncode)

def count(pattern: str, folder: Path):
    return len(list(folder.rglob(pattern)))

def main():
    print("========== FULL ADNI PIPELINE ==========")

    # Quick sanity checks
    if not ADNI_ROOT.exists():
        print(f"[FATAL] ADNI root not found: {ADNI_ROOT}")
        sys.exit(1)
    if not ADNIMERGE.exists():
        print(f"[FATAL] ADNIMERGE not found: {ADNIMERGE}")
        sys.exit(1)

    # 1) Convert all DICOMs → NIfTI
    run_py(SCRIPTS["convert"])
    n_nii = count("*.nii*", ALL_NIFTI)
    print(f"[OK] NIfTI count in ALL_NIFTI: {n_nii}")
    if n_nii == 0:
        print("[FATAL] No NIfTI created. Aborting.")
        sys.exit(1)

    # 2) Preprocess all NIfTIs
    run_py(SCRIPTS["preprocess"])
    n_pre = count("*_preproc.nii.gz", PREPROC)
    print(f"[OK] Preprocessed volumes: {n_pre}")
    if n_pre == 0:
        print("[FATAL] No preprocessed volumes. Aborting.")
        sys.exit(1)

    # 3) Extract features
    run_py(SCRIPTS["features"])
    feats_csv = FEATURES / "features.csv"
    if not feats_csv.exists():
        print(f"[FATAL] Missing features.csv at {feats_csv}")
        sys.exit(1)
    print(f"[OK] Features table: {feats_csv}")

    # 4) Merge with ADNIMERGE
    run_py(SCRIPTS["merge"])
    merged_csv = FEATURES / "merged_features.csv"
    if not merged_csv.exists():
        print(f"[FATAL] Missing merged_features.csv at {merged_csv}")
        sys.exit(1)
    print(f"[OK] Merged table: {merged_csv}")

    # 5) Train models
    #    (optional) multi-class baseline
    run_py(SCRIPTS["train_multi"])
    #    AD vs non-AD baseline (your focus)
    run_py(SCRIPTS["train_ad"])

    # 6) Score all with AD models
    run_py(SCRIPTS["score_ad"])
    pred_csv = FEATURES / "ad_predictions.csv"
    if pred_csv.exists():
        print(f"[OK] AD predictions: {pred_csv}")

    print("\n========== DONE ==========")
    print(f"ALL_NIFTI -> {ALL_NIFTI}")
    print(f"PREPROC    -> {PREPROC}")
    print(f"FEATURES   -> {FEATURES}")
    print(f"MODELS     -> {MODELS}")

if __name__ == "__main__":
    main()
