# preprocess_mri.py
# Robust MRI preprocessing for ADNI-style T1 volumes.
# - Reads NIfTI from ALL_NIFTI (or --in)
# - Casts to Float32 before N4 (required)
# - Builds a brain mask via Otsu + cleanup (UInt8)
# - N4 bias correction
# - Intensity normalization (robust [p1,p99] -> [0,1])
# - Writes *_preproc.nii.gz and *_mask.nii.gz into PREPROC (or --out)
# - Optional QC PNGs (skips gracefully if matplotlib is missing)

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# ---------------------------
# Utility Functions
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def itk_read(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return img


def itk_write(img: sitk.Image, path: Path) -> None:
    sitk.WriteImage(img, str(path))


def numpy_from_sitk(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    return arr


def sitk_from_numpy(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out


def otsu_brain_mask(img: sitk.Image) -> sitk.Image:
    """Generate a binary brain mask using Otsu + morphological cleanup.

    Parameters
    ----------
    img: sitk.Image (expected Float32)

    Returns
    -------
    sitk.Image (UInt8 mask)
    """
    # Slight smoothing to stabilize Otsu
    sm = sitk.CurvatureFlow(image1=img, timeStep=0.125, numberOfIterations=3)

    # Otsu threshold in 3D
    otsu = sitk.OtsuThreshold(sm, 0, 1, 200)

    # Keep largest component to remove neck/non-brain objects
    cc = sitk.ConnectedComponent(otsu)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    if stats.GetNumberOfLabels() == 0:
        return sitk.Cast(otsu, sitk.sitkUInt8)

    # Find largest label
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    mask = sitk.Equal(cc, int(largest_label))

    # Morphological closing + fill holes
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    mask = sitk.VotingBinaryHoleFilling(mask, radius=[2, 2, 2], majorityThreshold=1)
    return sitk.Cast(mask, sitk.sitkUInt8)


def n4_bias_correct(moving: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """Run N4 bias field correction on a Float32 image with a binary mask."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # Reasonable defaults for T1w
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-6)
    corrector.SetSplineOrder(3)

    corrected = corrector.Execute(moving, mask)
    corrected = sitk.Cast(corrected, sitk.sitkFloat32)
    return corrected


def robust_minmax(arr: np.ndarray, mask: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> Tuple[float, float]:
    vals = arr[mask > 0].astype(np.float32)
    if vals.size == 0:
        # fallback to global percentiles
        vals = arr.astype(np.float32).ravel()
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if hi <= lo:
        # avoid divide by zero
        hi = lo + 1e-6
    return float(lo), float(hi)


def normalize_intensity(img: sitk.Image, mask: sitk.Image) -> sitk.Image:
    arr = numpy_from_sitk(img)
    msk = numpy_from_sitk(mask)
    lo, hi = robust_minmax(arr, msk)
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return sitk_from_numpy(arr, img)


def make_qc_png(img: sitk.Image, mask: sitk.Image, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        # Silently skip if matplotlib missing
        return

    arr = numpy_from_sitk(img)
    msk = numpy_from_sitk(mask)

    # Choose middle slices
    zc = arr.shape[0] // 2
    yc = arr.shape[1] // 2
    xc = arr.shape[2] // 2

    fig = plt.figure(figsize=(10, 10))

    def _subplot(idx, plane, data, title):
        ax = fig.add_subplot(3, 3, idx)
        ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    # Axial
    _subplot(1, "ax", arr[zc, :, :], f"Axial z={zc}")
    _subplot(2, "ax", msk[zc, :, :], f"Mask z={zc}")
    _subplot(3, "ax", (arr[zc, :, :] * (msk[zc, :, :] > 0)), "Overlay")
    # Coronal
    _subplot(4, "cor", arr[:, yc, :], f"Coronal y={yc}")
    _subplot(5, "cor", msk[:, yc, :], f"Mask y={yc}")
    _subplot(6, "cor", (arr[:, yc, :] * (msk[:, yc, :] > 0)), "Overlay")
    # Sagittal
    _subplot(7, "sag", arr[:, :, xc], f"Sagittal x={xc}")
    _subplot(8, "sag", msk[:, :, xc], f"Mask x={xc}")
    _subplot(9, "sag", (arr[:, :, xc] * (msk[:, :, xc] > 0)), "Overlay")

    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


# ---------------------------
# Main Pipeline
# ---------------------------

def process_one(nifti_path: Path, out_dir: Path, qc_dir: Path, write_qc: bool = True) -> Tuple[Path, Path]:
    """Process a single NIfTI volume.

    Returns
    -------
    Tuple of (preproc_path, mask_path)
    """
    img = itk_read(nifti_path)

    # Cast to float32 before N4
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    # Initial mask
    mask = otsu_brain_mask(img_f)

    # N4 bias correction (use mask)
    n4 = n4_bias_correct(img_f, mask)

    # Renormalize inside mask to [0,1]
    norm = normalize_intensity(n4, mask)

    stem = nifti_path.name
    stem = stem.replace(".nii.gz", "").replace(".nii", "")

    preproc_path = out_dir / f"{stem}_preproc.nii.gz"
    mask_path = out_dir / f"{stem}_mask.nii.gz"

    ensure_dir(out_dir)
    itk_write(norm, preproc_path)
    itk_write(mask, mask_path)

    if write_qc:
        qc_png = qc_dir / f"{stem}_qc.png"
        make_qc_png(norm, mask, qc_png)

    return preproc_path, mask_path


def find_inputs(in_dir: Path, pattern: str) -> list[Path]:
    paths = []
    # Support both .nii and .nii.gz by default pattern
    for p in sorted(glob.glob(str(in_dir / pattern), recursive=False)):
        if p.lower().endswith((".nii", ".nii.gz")):
            paths.append(Path(p))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust MRI preprocessing for NIfTI volumes (ADNI-style T1w).")

    default_root = Path(r"C:\\Users\\tulikachoudhary\\Desktop\\c502")

    parser.add_argument("--root", type=Path, default=default_root, help="Project root containing ALL_NIFTI and PREPROC (default: %(default)s)")
    parser.add_argument("--in", dest="in_dir", type=Path, default=None, help="Input folder with NIfTI files (default: <root>/ALL_NIFTI)")
    parser.add_argument("--out", dest="out_dir", type=Path, default=None, help="Output folder for preprocessed files (default: <root>/PREPROC)")
    parser.add_argument("--qc", dest="qc_dir", type=Path, default=None, help="QC image folder (default: <root>/QC)")
    parser.add_argument("--pattern", type=str, default="*.nii*", help="Glob pattern for inputs (default: %(default)s)")
    parser.add_argument("--no-qc", action="store_true", help="Disable QC PNG generation")

    args = parser.parse_args()

    root: Path = args.root
    in_dir = args.in_dir if args.in_dir is not None else (root / "ALL_NIFTI")
    out_dir = args.out_dir if args.out_dir is not None else (root / "PREPROC")
    qc_dir = args.qc_dir if args.qc_dir is not None else (root / "QC")

    setattr(args, "in_dir", in_dir)
    setattr(args, "out_dir", out_dir)
    setattr(args, "qc_dir", qc_dir)

    return args


def main() -> int:
    args = parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    qc_dir: Path = args.qc_dir
    pattern: str = args.pattern
    write_qc: bool = not args.no_qc

    if not in_dir.exists():
        print(f"[ERROR] Input directory does not exist: {in_dir}", file=sys.stderr)
        return 2

    ensure_dir(out_dir)
    if write_qc:
        ensure_dir(qc_dir)

    inputs = find_inputs(in_dir, pattern)
    if len(inputs) == 0:
        print(f"[WARN] No NIfTI files found in {in_dir} matching '{pattern}'.", file=sys.stderr)
        return 1

    iterator = tqdm(inputs, desc="Preprocessing", unit="vol") if HAS_TQDM else inputs

    n_ok = 0
    n_fail = 0

    for p in iterator:
        try:
            preproc_path, mask_path = process_one(p, out_dir, qc_dir, write_qc)
            n_ok += 1
            if HAS_TQDM:
                iterator.set_postfix(ok=n_ok, fail=n_fail)
            else:
                print(f"[OK] {p.name} -> {preproc_path.name}, {mask_path.name}")
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {p}: {e}", file=sys.stderr)
            continue

    print(f"Done. Success: {n_ok}, Failed: {n_fail}, Inputs: {len(inputs)}")
    return 0 if n_ok > 0 and n_fail == 0 else (0 if n_ok > 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
