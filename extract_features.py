import os, glob
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.stats import skew, kurtosis

# Folders
IN_DIR  = r"C:\Users\tulikachoudhary\Desktop\c502\PREPROC"
OUT_DIR = r"C:\Users\tulikachoudhary\Desktop\c502\FEATURES"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "features.csv")

def robust_stats(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return med, q3 - q1

def load_array(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (z,y,x)
    spacing = img.GetSpacing()                              # (x,y,z)
    vox_vol = spacing[0] * spacing[1] * spacing[2]          # mm^3
    return arr, vox_vol

def three_class_otsu(arr):
    """
    SimpleITK's OtsuMultipleThresholds does not take a 'maskImage' kwarg.
    We apply the mask by zeroing out non-brain voxels, run 3-class Otsu,
    then set labels outside mask to -1.
    """
    mask = arr > 0
    if np.count_nonzero(mask) < 100:
        return None

    arr_masked = np.where(mask, arr, 0).astype(np.float32)
    img = sitk.GetImageFromArray(arr_masked)
    # 3 classes -> 2 thresholds
    otsu_img = sitk.OtsuMultipleThresholds(img, numberOfThresholds=2)
    labels = sitk.GetArrayFromImage(otsu_img).astype(np.int16)
    labels[~mask] = -1
    return labels

def features_for_file(nii_path):
    subj = os.path.splitext(os.path.basename(nii_path))[0].replace(".nii","")
    arr, vox_vol = load_array(nii_path)

    brain_mask = arr > 0
    brain_vals = arr[brain_mask]
    if brain_vals.size == 0:
        return None

    # Intensity stats (suppress scary warnings for nearly-constant regions)
    mean = float(np.mean(brain_vals))
    std  = float(np.std(brain_vals))
    med, iqr = [float(x) for x in robust_stats(brain_vals)]
    sk   = float(skew(brain_vals, bias=True))       # use bias=True to avoid precision warnings
    ku   = float(kurtosis(brain_vals, fisher=True, bias=True))

    icv_mm3 = float(brain_vals.size * vox_vol)
    icv_ml  = icv_mm3 / 1000.0

    # Quick 3-class Otsu for CSF/GM/WM proxies
    csf_ml = gm_ml = wm_ml = np.nan
    seg = three_class_otsu(arr)
    if seg is not None:
        # Rank labels by median intensity (low->high ≈ CSF, GM, WM for T1)
        med_by_lbl = []
        for lbl in [0, 1, 2]:
            vals = arr[(seg == lbl)]
            if vals.size:
                med_by_lbl.append((lbl, float(np.median(vals))))
        med_by_lbl.sort(key=lambda t: t[1])
        if len(med_by_lbl) == 3:
            csf_lbl, gm_lbl, wm_lbl = [t[0] for t in med_by_lbl]
            csf_ml = float(np.sum(seg == csf_lbl) * vox_vol / 1000.0)
            gm_ml  = float(np.sum(seg == gm_lbl)  * vox_vol / 1000.0)
            wm_ml  = float(np.sum(seg == wm_lbl)  * vox_vol / 1000.0)

    return {
        "subject": subj,
        "path": nii_path,
        "vox_mm3": vox_vol,
        "brain_voxels": int(brain_vals.size),
        "ICV_ml": icv_ml,
        "int_mean": mean,
        "int_std": std,
        "int_median": med,
        "int_IQR": iqr,
        "int_skew": sk,
        "int_kurtosis": ku,
        "CSF_ml": csf_ml,
        "GM_ml": gm_ml,
        "WM_ml": wm_ml
    }

def main():
    # Only use the *preprocessed* images, skip *_mask.nii.gz
    files = sorted(glob.glob(os.path.join(IN_DIR, "*_preproc.nii*")))
    print(f"Found {len(files)} preprocessed NIfTI files")
    rows = []
    for f in files:
        try:
            feat = features_for_file(f)
            if feat:
                rows.append(feat)
                print(f"OK  : {os.path.basename(f)}")
            else:
                print(f"SKIP: {os.path.basename(f)} (no brain voxels)")
        except Exception as e:
            print(f"FAIL: {f} -> {e}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(rows)} rows -> {OUT_CSV}")

if __name__ == "__main__":
    main()
