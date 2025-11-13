import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Step 1: Select one MRI file to visualize ---
file_path = Path(r"C:\Users\tulikachoudhary\Desktop\c502\ALL_NIFTI\002_Accelerated_Sagittal_MPRAGE_20230621135509_033_S_7079_033_S_7079.nii.gz")

# --- Step 2: Load MRI data ---
img = nib.load(str(file_path))
data = img.get_fdata()

print(f"Loaded MRI shape: {data.shape}")  # e.g., (256, 256, 196)

# --- Step 3: Normalize voxel intensity between 0–1 ---
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# --- Step 4: Show the middle slice of the brain ---
middle_slice = data[:, :, data.shape[2] // 2]

plt.imshow(middle_slice, cmap='gray')
plt.title(f"{file_path.name}\nMiddle slice")
plt.axis('off')
plt.show()
