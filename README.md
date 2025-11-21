## Alzheimer’s XAI codebase
Transparent, multimodal, explainable AI framework for Alzheimer’s Disease (AD) research using ADNI. This project fuses structural MRI with clinical, cognitive, and genetic features, trains a 3D-CNN + Bi-LSTM fusion model, and provides XAI overlays (SHAP, LIME, PDP, Grad-CAM) to interpret predictions. Optional human in the loop (HITL) feedback allows clinicians to refine the model over time.
Important: This code is for research only. It is not a medical device and must not be used for clinical diagnosis.
Project overview
This repository moves beyond black-box AD classifiers by pairing state-of-the-art multimodal modeling with explainability that clinicians can examine and challenge.
•	Multimodal fusion: T1-weighted MRI volumes via 3D-CNN combined with longitudinal clinical/genetic features via Bi-LSTM.
•	Explainability first: SHAP, LIME, PDP, and 3D Grad-CAM to visualize and quantify drivers of predictions.
•	Clinician feedback loop: HITL interface and retraining scripts to incorporate expert corrections and uncertainty flags.
•	Reproducible pipeline: End-to-end orchestration from MRI preprocessing to scoring and XAI export.
Data access and inputs
•	Source: Alzheimer’s Disease Neuroimaging Initiative (ADNI). Registration and approval are required to download data.
•	Modalities:
o	MRI: T1-weighted structural MRI in DICOM format, converted to NIfTI.
o	Clinical/cognitive: ADNI CSVs (e.g., MMSE, RAVLT, FAQ, PTDEMOG).
o	Genetics: SNP/APOE4 indicators from ADNI genetics tables.
•	Local setup: Adapt paths in scripts to your ADNI directory structure. Raw ADNI files are not stored in this repo.
Installation and environment
•	Requirements:
o	Python: 3.10+
o	GPU: CUDA-enabled GPU recommended for 3D-CNN training and Grad-CAM
o	Libraries: numpy, pandas, scikit-learn, torch or tensorflow, nibabel/SimpleITK, matplotlib, seaborn, shap, lime
•	Setup:
o	Create environment: Use your preferred virtual environment tool.
o	Install dependencies:
bash
pip install -r requirements.txt
o	Configure paths: Update config files or script arguments to point to your ADNI folders.

Steps to run the project :

# 1) Convert and preprocess MRI
# DICOM → NIfTI conversion
./dcm2niix -o ./nifti_output ./dicom_input

python convert_adni_dicom_to_nifti.py --input ./dicom_input --output ./nifti_output
python convert_dicom_batch.py --root ./adni_dicom --out ./adni_nifti

# MRI preprocessing
python preprocess_mri.py --input ./adni_nifti --output ./preprocessed_mri

# Visualization
python view_mri_slice.py --file ./preprocessed_mri/subj01.nii.gz
python viz_topk_slices.py --model ./trained_model.pth --input ./preprocessed_mri/subj01.nii.gz

# 2) Build clinical and genetic feature tables
python merge_clinical.py --tables ./clinical_csvs --out ./merged_features.csv
python merge_features_with_adni.py --input ./clinical_csvs --output ./features.csv
python diag_convert.py --input ./features.csv --output ./features_diag.csv
python eda_features.py --input ./features_diag.csv
python extract_features.py --input ./features_diag.csv --output ./final_features.csv

# 3) Link MRI and clinical data
python ensure_ad_links_from_path_tokens.py --mri ./preprocessed_mri --clinical ./final_features.csv
python link_ad_by_site_and_date.py --mri ./preprocessed_mri --clinical ./final_features.csv
python force_link.py --mri ./preprocessed_mri --clinical ./final_features.csv
python row_confusion.py --clinical ./final_features.csv

# 4) Train models
python train_baseline.py --features ./final_features.csv
python train_ad_baseline.py --features ./final_features.csv

python train_multimodal_ai.py --mri ./preprocessed_mri --features ./final_features.csv
python train_multimodal_ptid_split.py --mri ./preprocessed_mri --features ./final_features.csv
python train_multimodal_ptid_strat.py --mri ./preprocessed_mri --features ./final_features.csv

python run_full_pipeline.py --config ./config.yaml

# 5) Evaluate and score
python score_all_ad.py --model ./trained_model.pth --data ./test_data
python score_multimodal_all.py --model ./trained_model.pth --mri ./preprocessed_mri --features ./final_features.csv
python topk_confident_ad.py --predictions ./preds.csv --k 10

# 6) Explainability (XAI)
python make_shap.py --model ./trained_model.pth --features ./final_features.csv
python make_shap_tabular.py --model ./trained_model.pth --features ./final_features.csv
python make_shap_mri.py --model ./trained_model.pth --mri ./preprocessed_mri

python make_lime_pdp_tabular.py --model ./trained_model.pth --features ./final_features.csv
python gradcam_3d.py --model ./trained_model.pth --mri ./preprocessed_mri/subj01.nii.gz
python xai_master.py --model ./trained_model.pth --mri ./preprocessed_mri --features ./final_features.csv

# 7) Human-in-the-loop feedback
# Launch feedback form
open feedback_form.html

# Inspect feedback
python peek_fb.py --db ./hitl.db
python pending_python.py --db ./hitl.db

# Retrain with feedback
python retrain_fb.py --db ./hitl.db --mri ./preprocessed_mri --features ./final_features.csv

