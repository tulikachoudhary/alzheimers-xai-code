# Alzheimer’s-XAI-Codebase

A multimodal, explainable AI framework for Alzheimer’s Disease (AD) research using the ADNI dataset.  
This toolkit fuses MRI, clinical, cognitive, and genetic features, trains a 3D‑CNN + Bi‑LSTM fusion model, and provides explainability overlays (SHAP, LIME, PDP, Grad‑CAM). It also supports an optional human‑in‑the‑loop (HITL) feedback loop for clinician review and iterative retraining.

---

## Quick summary
- Converts ADNI MRI scans (DICOM → NIfTI) and preprocesses volumes.
- Merges clinical, demographic, cognitive, and genetic features into tabular datasets.
- Trains multimodal models (3D‑CNN for MRI + Bi‑LSTM for tabular features).
- Provides explainability overlays (SHAP, LIME, PDP, Grad‑CAM).
- Supports clinician feedback (HITL) to refine predictions and retrain models.

---

## Contents (high level)
- `convert_adni_dicom_to_nifti.py`, `convert_dicom_batch.py` — DICOM → NIfTI conversion utilities.
- `preprocess_mri.py` — MRI preprocessing (resizing, cropping, normalization).
- `merge_clinical.py`, `merge_features_with_adni.py`, `extract_features.py` — Clinical/genetic feature merging and engineering.
- `ensure_ad_links_from_path_tokens.py`, `link_ad_by_site_and_date.py` — Linking MRI scans with clinical rows.
- `train_baseline.py`, `train_multimodal_ai.py`, `lstm_fusion.py` — Baseline and multimodal training scripts.
- `score_all_ad.py`, `score_multimodal_all.py` — Evaluation and scoring utilities.
- `make_shap.py`, `make_lime_pdp_tabular.py`, `gradcam_3d.py`, `xai_master.py` — Explainability scripts.
- `feedback_form.html`, `hitl.db`, `retrain_fb.py` — Human‑in‑the‑loop feedback interface and retraining.

---

## Requirements
- Python 3.10+ recommended (tested on Python 3.10–3.11).
- CUDA‑enabled GPU recommended for 3D‑CNN training and Grad‑CAM.
- The provided `requirements.txt` contains core dependencies. Additional packages may be needed.

Recommended Python packages (installed via pip):
- numpy
- pandas
- scikit-learn
- torch or tensorflow (depending on training scripts)
- nibabel, SimpleITK
- matplotlib, seaborn
- shap, lime

Start with the included `requirements.txt` and add extras as needed.

---

## Quickstart (Windows PowerShell)


1. Create and activate a virtual environment


```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# additional packages used by some scripts
pip install nibabel SimpleITK matplotlib seaborn shap lime

```

2. Install dependencies


```powershell
pip install -r requirements.txt
# additional packages used by some scripts
pip install nibabel SimpleITK matplotlib seaborn shap lime
Convert and preprocess MRI

```
3. Convert and preprocess MRI


```powershell
# DICOM → NIfTI
.\dcm2niix.exe -o .\nifti_output .\dicom_input
python .\convert_adni_dicom_to_nifti.py --input .\dicom_input --output .\nifti_output
python .\convert_dicom_batch.py --root .\adni_dicom --out .\adni_nifti

```
4. Build clinical and genetic feature tables

```powershell
# Preprocess MRI
python .\preprocess_mri.py --input .\adni_nifti --output .\preprocessed_mri

#Build clinical and genetic feature tables


powershell
python .\merge_clinical.py --tables .\clinical_csvs --out .\merged_features.csv
python .\merge_features_with_adni.py --input .\clinical_csvs --output .\features.csv
python .\diag_convert.py --input .\features.csv --output .\features_diag.csv
python .\extract_features.py --input .\features_diag.csv --output .\final_features.csv

```

5. Link MRI and clinical data


```powershell
python .\ensure_ad_links_from_path_tokens.py --mri .\preprocessed_mri --clinical .\final_features.csv
python .\link_ad_by_site_and_date.py --mri .\preprocessed_mri --clinical .\final_features.csv
Train models

```

6. Train models


```powershell
python .\train_baseline.py --features .\final_features.csv
python .\train_multimodal_ai.py --mri .\preprocessed_mri --features .\final_features.csv
python .\run_full_pipeline.py --config .\config.yaml
Evaluate and score

```

7. Evaluate and score


```powershell
python .\score_all_ad.py --model .\trained_model.pth --data .\test_data
python .\score_multimodal_all.py --model .\trained_model.pth --mri .\preprocessed_mri --features .\final_features.csv
Run explainability overlays

```

8. Run explainability overlays


```powershell
python .\make_shap.py --model .\trained_model.pth --features .\final_features.csv
python .\make_lime_pdp_tabular.py --model .\trained_model.pth --features .\final_features.csv
python .\gradcam_3d.py --model .\trained_model.pth --mri .\preprocessed_mri\subj01.nii.gz
Human‑in‑the‑loop feedback

```

9. Human‑in‑the‑loop feedback


```powershell
# Launch feedback form
start .\feedback_form.html

# Inspect feedback
python .\peek_fb.py --db .\hitl.db

# Retrain with feedback
python .\retrain_fb.py --db .\hitl.db --mri .\preprocessed_mri --features .\final_features.csv


```

# **Project Workflow Overview**

- Converts ADNI MRI scans (DICOM → NIfTI) and preprocesses volumes.

- Merges clinical, cognitive, demographic, and genetic tables into tabular features.

- Aligns MRI scans with clinical rows by subject ID and exam date.

- Trains baseline tabular models and multimodal fusion models (3D‑CNN + Bi‑LSTM).

- Evaluates models and exports predictions.

- Applies explainability methods (SHAP, LIME, PDP, Grad‑CAM) to interpret predictions.

- Uses HITL feedback to refine labels and retrain models iteratively.

---

## **Notes on mismatched or missing dependencies/files**

- The top‑level requirements.txt contains core packages but may omit extras (e.g., shap, lime, nibabel).

- Some scripts may reference either PyTorch or TensorFlow — install both if needed.

- Ensure ADNI data paths are correctly set; raw data is not included in this repo.

- HITL feedback requires SQLite (hitl.db) and the provided HTML form.
