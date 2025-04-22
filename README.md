## 🧠 Functional Connectivity Extraction

This module computes functional connectivity matrices from resting-state fMRI data using the [Schaefer 2018 atlas](https://www.nitrc.org/projects/schaefer_atlas/). Each matrix is Fisher z-transformed and saved in a 3D NumPy array of shape *(n_rois × n_rois × n_subjects)*.

### 🔄 Workflow
- Load preprocessed rs-fMRI data for each subject
- Resample each subject’s image to match the atlas resolution
- Extract region-wise time series using `NiftiLabelsMasker`
- Compute correlation-based functional connectivity
- Apply Fisher z-transformation
- Save the matrices to disk as a `.npy` file

### 📁 Script
[`extract_connectivity_matrices.py`](./extract_connectivity_matrices.py)

### 🚀 How to Run

```bash
python extract_connectivity_matrices.py
```

You can customize:
- `subjects_location`: Path to fMRI subjects (each in its own folder with a `.nii.gz` file)
- `output_path`: Output filename for the resulting `.npy` connectivity matrix
- `n_rois`: Number of Schaefer atlas ROIs (e.g., 100, 200, 400)

---

## 🧪 Unit Tests

Basic unit tests are provided to validate:
- Initialization of the masker
- Output shape and validity of the connectivity matrix
- Handling of simple dummy 4D fMRI data

### 📁 Test File
[`tests/test_extract_connectivity.py`](./tests/test_extract_connectivity.py)

### ▶️ Run the tests

```bash
pytest tests/
```

### 📦 Test Dependencies

Make sure you install the following before running tests:

```bash
pip install pytest nilearn numpy nibabel
```

---
