"""The segmentation extractor reads 0/255 masks as binary, one region per component."""

import warnings

import numpy as np
import pytest

from defect_clustering.features.segmentation_features import (
    N_BASE_FEATURES,
    extract_region_features,
    extract_segmentation_features,
)


def _two_blobs(value):
    """A mask with two separate rectangles of different sizes."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:8, 2:8] = value       # 36 px
    mask[20:35, 15:35] = value   # 300 px
    return mask


@pytest.mark.parametrize("value", [1, 255])
def test_each_component_of_a_binary_mask_is_a_region(value):
    feats = extract_region_features(_two_blobs(value))
    assert feats.shape == (2, N_BASE_FEATURES)
    assert sorted(feats[:, 0]) == [36.0, 300.0]   # column 0 is the area


def test_0_255_and_boolean_masks_give_the_same_features():
    np.testing.assert_array_equal(extract_region_features(_two_blobs(255)),
                                  extract_region_features(_two_blobs(255) > 0))


def test_mean_and_max_differ_when_the_regions_differ():
    vec = extract_segmentation_features(_two_blobs(255), aggregation="mean_max")
    mean, mx = vec[:N_BASE_FEATURES], vec[N_BASE_FEATURES:]
    assert (mean[0], mx[0]) == (168.0, 300.0)


def test_diagonal_pixels_form_one_region():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4, 4] = mask[5, 5] = 255     # touching by a corner only: 8-connected
    assert extract_region_features(mask).shape == (1, N_BASE_FEATURES)


def test_a_label_image_keeps_its_labels():
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[2:8, 2:8] = 2
    mask[30:35, 30:35] = 2       # label 2 spans two separate pieces
    mask[15:25, 15:25] = 5
    feats = extract_region_features(mask)
    assert sorted(feats[:, 0]) == [61.0, 100.0]


def test_no_deprecated_regionprops_property_is_read():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        extract_region_features(_two_blobs(255), intensity_image=_two_blobs(255))
