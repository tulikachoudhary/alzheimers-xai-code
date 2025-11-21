# Alzheimer’s XAI Codebase

Multimodal, explainable AI framework for **Alzheimer’s Disease (AD) diagnosis** using the **ADNI** dataset.  
This repo contains the full pipeline:

- MRI (DICOM → NIfTI) preprocessing  
- Clinical + demographic + cognitive feature merging  
- Multimodal **3D-CNN + Bi-LSTM** model for AD vs non-AD  
- **Explainable AI (XAI)** overlays: SHAP, LIME, PDP, Grad-CAM  
- Optional **human-in-the-loop (HITL) clinician feedback** to refine the model

> ⚠️ **Note:** This code is for **research** only. It is **not** a medical device and must not be used for clinical diagnosis.

---

## 1. Project Overview

The goal of this project is to move beyond black-box AD classifiers and build a **transparent, multi-modal framework** that clinicians can actually trust.

We:

1. Fuse **MRI**, **clinical**, and **genetic** information from **ADNI**.  
2. Train a **3D Convolutional Neural Network (3D-CNN)** on T1-weighted MRI volumes.  
3. Train a **Bidirectional Long Short-Term Memory network (Bi-LSTM)** on longitudinal clinical + genetic features.  
4. Combine these branches in a **multimodal fusion** model for probabilistic AD prediction.  
5. Apply **XAI methods** (SHAP, LIME, PDP, Grad-CAM) to understand *why* the model predicts AD.  
6. Use an optional **clinician feedback loop** (HITL) to correct mistakes and iteratively retrain the model.

---

## 2. Data

This project uses data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**.

- Website: https://adni.loni.usc.edu  
- You must **register and be approved by ADNI** to download the data.  
- Raw data (MRI, clinical CSVs, genetics) are **not** stored in this repo.

Typical inputs:

- **MRI**: T1-weighted structural MRI in DICOM format  
- **Clinical / cognitive**: CSVs such as `MMSE.csv`, `RAVLT.csv`, `FAQ.csv`, `PTDEMOG.csv`, etc.  
- **Genetic**: SNP / APOE4 indicators (from ADNI genetics tables)

You’ll need to adapt paths in the scripts to your local ADNI directory structure.

---

## 3. Repository Structure

Key files and what they do (grouped by function):

### 3.1. DICOM → NIfTI Conversion & MRI Preprocessing

- `dcm2niix.exe`  
  - External binary used for DICOM → NIfTI conversion.

- `convert_adni_dicom_to_nifti.py`  
- `convert_dicom_batch.py`  
  - Utilities to batch-convert ADNI DICOM folders into NIfTI volumes and organize them by subject / visit.

- `preprocess_mri.py`  
  - Preprocessing pipeline for NIfTI MRI (e.g., resizing, cropping, intensity normalization, orientation).

- `view_mri_slice.py`  
- `viz_topk_slices.py`  
  - Helper scripts to visualize MRI slices and CAM overlays for inspection.

---

### 3.2. Clinical / Tabular Feature Engineering & Merging

- `MMSE.csv`, `RAVLT.csv`, `FAQ.csv`, `PTDEMOG.csv`, etc.  
  - Raw ADNI clinical / demographic tables (downloaded separately from ADNI).

- `merge_clinical.py`  
- `merge_features_with_adni.py`  
- `merge_by_row.py`, `merge_row.py`  
  - Merge multiple ADNI tables into a **single feature table** keyed by subject ID and visit.

- `diag_convert.py`  
- `eda_features.py`  
  - Convert diagnosis labels, run exploratory data analysis, and engineer additional features.

- `extract_features.py`  
  - Build the final tabular feature matrix (e.g., cognitive scores, volumes, APOE4) used by the LSTM and XAI tools.

---

### 3.3. Linking MRI with Clinical Data

- `ensure_ad_links_from_path_tokens.py`  
- `link_ad_by_site_and_date.py`  
- `force_link.py`  
  - Scripts to **align MRI scans with the correct clinical rows** via subject ID, site, and exam date.

- `row_confusion.py`  
  - Diagnostics for mismatched or ambiguous ID/date rows.

---

### 3.4. Model Training (Baseline + Multimodal)

- `train_baseline.py`, `train_ad_baseline.py`  
  - Baseline tabular models (e.g., classical ML) using clinical features only.

- `train_model.py`  
  - Core training logic shared across models.

- `train_multimodal_ai.py`  
- `train_multimodal_ptid_split.py`  
- `train_multimodal_ptid_strat.py`  
  - Train the **multimodal 3D-CNN + Bi-LSTM** model with subject-wise splits and/or stratified sampling.

- `lstm_fusion.py`  
  - Fusion architecture defining how MRI and tabular features are combined.

- `run_full_pipeline.py`  
  - Orchestrates end-to-end runs (preprocessing → training → evaluation), depending on configuration.

---

### 3.5. Scoring & Evaluation

- `score_all_ad.py`  
- `score_multimodal_all.py`  
- `topk_confident_ad.py`  
  - Score all (or subsets of) subjects and export predictions, including **top-K high-confidence AD candidates**.

- `train_baseline.py`, `train_ad_baseline.py`  
  - Baseline metrics for comparison (accuracy, precision, recall, F1, ROC-AUC, etc.).

---

### 3.6. XAI (SHAP, LIME, PDP, Grad-CAM)

- `make_shap.py`  
- `make_shap_tabular.py`  
- `make_shap_mri.py`  
  - Generate SHAP values for tabular features and/or MRI components.

- `make_lime_pdp_tabular.py`  
  - Create **LIME explanations** and **Partial Dependence Plots (PDP)** for clinical/genetic features.

- `gradcam_3d.py`  
  - Compute **3D Grad-CAM** heatmaps over MRI volumes to visualize regions driving the CNN predictions.

- `xai_master.py`  
  - High-level driver script to run multiple XAI methods and consolidate outputs.

---

### 3.7. Human-in-the-Loop (Clinician Feedback)

- `feedback_form.html`  
  - Simple HTML interface for a clinician to review predictions and flag incorrect or uncertain cases.

- `hitl.db`  
  - SQLite database storing feedback (e.g., corrected labels, comments).

- `peek_fb.py`  
- `pending_python.py`  
- `retrain_fb.py`  
  - Tools to inspect feedback records and **retrain the model** based on clinician-validated labels.

---

## 4. Getting Started

### 4.1. Environment

Recommended:

- Python 3.10+  
- CUDA-enabled GPU (for 3D-CNN training and MRI Grad-CAM)  
- Typical libraries:  
  - `numpy`, `pandas`, `scikit-learn`  
  - `torch` or `tensorflow` (check the scripts for the framework used)  
  - `nibabel`, `SimpleITK` or similar for NIfTI  
  - `matplotlib`, `seaborn` for plots  
  - `shap`, `lime` for XAI

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
