import os
import numpy as np
import pytest
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
from nilearn import image
import nibabel as nib

from extract_connectivity_matrices import extract_connectivity, create_masker


@pytest.fixture(scope="module")
def setup_masker():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
    masker = create_masker(atlas.maps)
    return masker, atlas.maps


def create_dummy_fmri(subject_id, shape=(2, 2, 2, 10), output_dir="tests/mock_subject"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.random.rand(*shape)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    filepath = os.path.join(output_dir, f"{subject_id}.nii.gz")
    nib.save(img, filepath)
    return filepath, output_dir


def test_masker_initialization(setup_masker):
    masker, _ = setup_masker
    assert isinstance(masker, NiftiLabelsMasker)


def test_extract_connectivity_valid_output(setup_masker):
    masker, atlas_img = setup_masker
    subject_id = "test_subject"
    fmri_file, subject_path = create_dummy_fmri(subject_id)

    # Rename the file to match the extract_connectivity() expectation
    expected_path = os.path.join(subject_path, f"{os.path.basename(subject_path)}.gz")
    os.rename(fmri_file, expected_path)

    try:
        conn_matrix = extract_connectivity(subject_path, atlas_img, masker)
        assert conn_matrix.shape[0] == conn_matrix.shape[1]
        assert np.all(np.isfinite(conn_matrix))
        assert not np.all(conn_matrix == 0)
    finally:
        # Clean up
        os.remove(expected_path)
        os.rmdir(subject_path)
