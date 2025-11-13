# convert_dicom_batch.py
import os, subprocess, re, glob

ROOT = r"C:\Users\tulikachoudhary\Desktop\c502"
ADNI_ROOT = os.path.join(ROOT, "ADNI")
OUT_DIR = os.path.join(ROOT, "ALL_NIFTI")
DCM2NIIX = os.path.join(ROOT, "dcm2niix.exe")  # adjust if needed

os.makedirs(OUT_DIR, exist_ok=True)

ptid_re = re.compile(r"(\d{3}_S_\d{4})")

def find_series_dirs():
    series = []
    for dirpath, dirnames, filenames in os.walk(ADNI_ROOT):
        # DICOM leaf dir usually has many .dcm files
        if any(fn.lower().endswith(".dcm") for fn in filenames):
            series.append(dirpath)
    return series

def infer_ptid_from_path(path):
    m = ptid_re.search(path)
    return m.group(1) if m else None

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

series_dirs = find_series_dirs()
print(f"[INFO] Found {len(series_dirs)} series folders to convert.\n")

for sdir in series_dirs:
    # Try to get a stable SeriesDescription & Date from the parent folders
    ptid = infer_ptid_from_path(sdir) or "UNK"
    # Use leaf folder name as fallback descriptor
    leaf = os.path.basename(sdir)
    # Also grab two parents above to capture protocol/date folders
    parent1 = os.path.basename(os.path.dirname(sdir))
    parent2 = os.path.basename(os.path.dirname(os.path.dirname(sdir)))

    # Compose a clear filename: {parent2}_{parent1}_{leaf}_{PTID}
    fname = f"{parent2}_{parent1}_{leaf}_{ptid}"
    fname = safe_name(fname)

    print(f"[CONVERT] {sdir}")
    # dcm2niix will create: OUT_DIR\{fname}.nii.gz and .json
    cmd = [
        DCM2NIIX,
        "-b", "y",           # write BIDS-ish JSON
        "-z", "y",           # gz compress
        "-f", fname,         # our constructed filename with PTID
        "-o", OUT_DIR,       # output folder
        sdir
    ]
    subprocess.run(cmd, check=True)
    print()

# Summary
nii = glob.glob(os.path.join(OUT_DIR, "*.nii.gz"))
print(f"[SUMMARY] NIfTI files in {OUT_DIR}: {len(nii)}")
print(f"[OK] NIfTI count in ALL_NIFTI: {len(nii)}")
