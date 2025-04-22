"""
Extract functional connectivity matrices from resting-state fMRI data using the Schaefer 2018 atlas.

This script processes a directory of fMRI images, applies the Schaefer atlas,
computes correlation-based functional connectivity matrices for each subject,
and saves them as a 3D numpy array.

Author: Naga Thovinakere
"""

import os
import numpy as np
import pandas as pd
from nilearn import datasets, image, connectome
from nilearn.maskers import NiftiLabelsMasker
from sklearn.preprocessing import StandardScaler


def load_schaefer_atlas(n_rois=100):
    """Load the Schaefer 2018 parcellation atlas with the desired number of ROIs."""
    print(f"Fetching Schaefer atlas with {n_rois} ROIs...")
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    return atlas.maps, atlas.labels


def create_masker(atlas_filename):
    """Initialize a NiftiLabelsMasker using the Schaefer atlas."""
    print("Creating masker...")
    return NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                             memory='nilearn_cache', verbose=0)


def extract_connectivity(subject_path, atlas_filename, masker):
    """Extract z-transformed correlation matrix for a single subject."""
    print(f"Processing subject: {subject_path}")
    fmri_file = os.path.join(subject_path, os.path.basename(subject_path) + ".gz")

    # Resample image to atlas space for consistency
    resampled_img = image.resample_to_img(fmri_file, atlas_filename)

    # Extract and normalize time series
    time_series = masker.fit_transform(resampled_img)
    scaler = StandardScaler().fit(time_series)
    time_series_z = scaler.transform(time_series)

    # Compute correlation matrix
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    corr_matrix = correlation_measure.fit_transform([time_series_z])[0]

    # Apply Fisher z-transformation
    return np.arctanh(corr_matrix)


def main(subjects_dir, output_file, n_rois=100):
    """Main processing function."""
    atlas_filename, labels = load_schaefer_atlas(n_rois=n_rois)
    masker = create_masker(atlas_filename)

    subject_ids = sorted(os.listdir(subjects_dir))
    n_subjects = len(subject_ids)
    all_matrices = np.empty((n_rois, n_rois, n_subjects), dtype=np.float32)

    for i, subject_id in enumerate(subject_ids):
        subject_path = os.path.join(subjects_dir, subject_id)
        try:
            all_matrices[:, :, i] = extract_connectivity(subject_path, atlas_filename, masker)
        except Exception as e:
            print(f"Skipping subject {subject_id} due to error: {e}")
            continue

    np.save(output_file, all_matrices)
    print(f"\n Saved connectivity matrices to: {output_file}")


if __name__ == "__main__":
    # Define paths
    subjects_location = "/home/nagatv11/scratch/temp/01"
    output_path = "/home/nagatv11/projects/def-mgeddes/nagatv11/ukbb_mvpa_prediction/data/corr_mat_01_keep.npy"

    # Run the pipeline
    main(subjects_location, output_path, n_rois=100)
