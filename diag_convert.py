from pathlib import Path
import os

INPUT_ROOT  = Path(r"C:\Users\tulikachoudhary\Desktop\c502\ADNI")
OUTPUT_ROOT = Path(r"C:\Users\tulikachoudhary\Desktop\c502\nifti")

print("=== DIAG START ===")
print("INPUT_ROOT exists:", INPUT_ROOT.exists(), INPUT_ROOT)
print("OUTPUT_ROOT will be:", OUTPUT_ROOT)

if not INPUT_ROOT.exists():
    print("[ERROR] INPUT_ROOT not found. Create folder or update path in script.")
    raise SystemExit(1)

count_dirs = 0
for root, dirs, files in os.walk(INPUT_ROOT):
    if any(f.lower().endswith(".dcm") for f in files):
        count_dirs += 1
        if count_dirs <= 5:
            print("found DICOM folder:", root)
print("total DICOM-containing folders seen:", count_dirs)
print("=== DIAG END ===")
