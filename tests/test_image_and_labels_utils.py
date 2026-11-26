import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates, shift
from skimage import feature, filters, segmentation

from eigenp_utils.image_and_labels_utils import (
    create_ellipsoid_struct,
    estimate_inter_label_distance,
    fit_plane_ransac,
    generate_morphological_surface_mask,
    generate_plane_basis,
    optimized_entire_labels_touching_mask,
    sample_intensity_along_surface_normals,
    sample_intensity_around_points,
    sample_volume_plane,
    voronoi_otsu_labeling,
    windowed_slice_projection,
)

ISO_3D = {"Z": 1.0, "Y": 1.0, "X": 1.0}
ISO_2D = {"Y": 1.0, "X": 1.0}


def _blob_volume(shape, centers, sigma, amplitude=1000.0):
    """Create a synthetic volume with Gaussian blobs placed at `centers`."""
    vol = np.zeros(shape, dtype=np.float64)
    for c in centers:
        vol[tuple(c)] = 1.0
    vol = gaussian_filter(vol, sigma)
    return vol / vol.max() * amplitude


def test_voronoi_otsu_labeling():
    """Verify 2D isotropic and 3D anisotropic Voronoi-Otsu segmentation of two spots.

    Strategy: synthesize two well-separated Gaussian spots with seeded low-level
    background noise in 2D and in anisotropic 3D (`Z=2.0, Y=0.5, X=0.5`), then
    segment with scalar and tuple sigmas.
    Expected outcome: both runs produce exactly 2 distinct positive labels whose
    spot centers carry different non-zero IDs.
    """
    rng = np.random.default_rng(42)

    img_2d = np.zeros((20, 20), dtype=float)
    for center, scale in [((5, 5), 3), ((15, 15), 3)]:
        y, x = np.ogrid[-center[0] : 20 - center[0], -center[1] : 20 - center[1]]
        img_2d += np.exp(-(x**2 + y**2) / (2 * scale**2)) * 10
    img_2d += rng.random((20, 20)) * 0.1

    labels_2d = voronoi_otsu_labeling(img_2d, spot_sigma=1, outline_sigma=1)
    assert labels_2d.shape == (20, 20)
    assert len(np.unique(labels_2d[labels_2d > 0])) == 2
    assert labels_2d[5, 5] > 0 and labels_2d[15, 15] > 0
    assert labels_2d[5, 5] != labels_2d[15, 15]

    img_3d = np.zeros((10, 20, 20), dtype=float)
    for center, scale in [((5, 5, 5), 2), ((5, 15, 15), 2)]:
        z, y, x = np.ogrid[
            -center[0] : 10 - center[0],
            -center[1] : 20 - center[1],
            -center[2] : 20 - center[2],
        ]
        img_3d += np.exp(-(x**2 + y**2 + z**2) / (2 * scale**2)) * 10
    img_3d += rng.random((10, 20, 20)) * 0.1

    spacing = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    labels_3d = voronoi_otsu_labeling(
        img_3d, spot_sigma=(2, 1, 1), outline_sigma=1, pixel_sizes=spacing
    )
    assert labels_3d.shape == (10, 20, 20)
    assert len(np.unique(labels_3d[labels_3d > 0])) == 2
    assert labels_3d[5, 5, 5] > 0 and labels_3d[5, 15, 15] > 0
    assert labels_3d[5, 5, 5] != labels_3d[5, 15, 15]


def test_voronoi_otsu_translation_invariance():
    """Verify binary foreground segmentation of `voronoi_otsu_labeling` is translation-invariant.

    Strategy: shift a synthetic 2D Gaussian spot by an integer offset using
    `scipy.ndimage.shift`, segment both images, shift the mask back, and compare
    the interior region.
    Expected outcome: foreground overlap exceeds 95% inside the unclipped interior.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((30, 30))
    y, x = np.ogrid[-10:20, -15:15]
    img += np.exp(-(x**2 + y**2) / (2 * 2**2)) * 10
    img += rng.random((30, 30)) * 0.1

    shift_vec = (5, -3)
    img_shifted = shift(img, shift_vec, order=1, mode="constant", cval=0.0)

    labels_original = voronoi_otsu_labeling(img, spot_sigma=1, outline_sigma=1)
    labels_shifted = voronoi_otsu_labeling(img_shifted, spot_sigma=1, outline_sigma=1)

    mask_original = (labels_original > 0).astype(int)
    mask_shifted = (labels_shifted > 0).astype(int)
    mask_shifted_back = shift(
        mask_shifted, (-shift_vec[0], -shift_vec[1]), order=0, mode="constant", cval=0
    )

    overlap_ratio = np.mean(mask_original[5:-5, 5:-5] == mask_shifted_back[5:-5, 5:-5])
    assert overlap_ratio > 0.95, f"Translation invariance failed, overlap ratio was {overlap_ratio}"


def test_voronoi_otsu_intensity_scale_invariance():
    """Verify uniform positive intensity scaling does not alter Voronoi-Otsu boundaries.

    Strategy: run `voronoi_otsu_labeling` on a seeded noisy spot image and on
    `img * 5.0`, comparing the resulting binary foreground masks.
    Expected outcome: both masks are bit-for-bit identical.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((30, 30))
    y, x = np.ogrid[-15:15, -15:15]
    img += np.exp(-(x**2 + y**2) / (2 * 3**2)) * 10
    img += rng.random((30, 30)) * 0.1

    labels_base = voronoi_otsu_labeling(img, spot_sigma=1, outline_sigma=1)
    labels_scaled = voronoi_otsu_labeling(img * 5.0, spot_sigma=1, outline_sigma=1)
    assert np.array_equal(labels_base > 0, labels_scaled > 0)


def test_voronoi_otsu_deprecated_spacing_matches_pixel_sizes():
    """Verify deprecated `spacing` tuple parameter emits `FutureWarning` and matches `pixel_sizes`.

    Strategy: segment a 3D blob volume with `spacing=(2.0, 0.5, 0.5)` and with
    `pixel_sizes={'Z': 2.0, 'Y': 0.5, 'X': 0.5}`.
    Expected outcome: `spacing` emits a `FutureWarning` naming 0.1.0 (D08; it was a
    `DeprecationWarning`, hidden when attributed to library code) and yields the
    exact same label array as `pixel_sizes`.
    """
    vol = _blob_volume((10, 30, 30), [(5, 15, 15)], sigma=(1.0, 3.0, 3.0))
    with pytest.warns(FutureWarning, match=r"spacing.*0\.1\.0.*pixel_sizes"):
        a = voronoi_otsu_labeling(vol, 1, 1, spacing=(2.0, 0.5, 0.5))
    b = voronoi_otsu_labeling(vol, 1, 1, pixel_sizes={"Z": 2.0, "Y": 0.5, "X": 0.5})
    np.testing.assert_array_equal(a, b)


def test_voronoi_keeps_nuclei_in_thin_stacks_and_at_borders():
    """Verify border seeds and nuclei in thin 3D stacks are retained (`exclude_border=False`).

    Strategy: segment a 6-slice 3D stack with 4 nuclei and a 2D image with 3
    nuclei within 1-2 px of image borders.
    Expected outcome: all 4 3D nuclei and all 3 border 2D nuclei receive distinct labels.
    """
    centers = [(3, 15, 15), (3, 15, 45), (2, 45, 15), (3, 45, 45)]
    vol = _blob_volume((6, 60, 60), centers, sigma=(1.0, 3.0, 3.0))
    labels = voronoi_otsu_labeling(vol, spot_sigma=1, outline_sigma=1)
    assert len(np.unique(labels[labels > 0])) == 4

    img = _blob_volume((64, 64), [(1, 30), (32, 62), (62, 5)], sigma=3.0)
    labels_2d = voronoi_otsu_labeling(img, spot_sigma=1, outline_sigma=1)
    for c in [(1, 30), (32, 62), (62, 5)]:
        assert labels_2d[c] > 0
    assert len(np.unique(labels_2d[labels_2d > 0])) == 3


def test_voronoi_seed_spacing_is_physical():
    """Verify peak suppression distance is measured in physical units (3 finest-axis pixels).

    Strategy: place two Z-stacked nuclei 6 um (3 slices at Z=2 um) apart and
    compare with two XY-adjacent nuclei 1 um (2 px at XY=0.5 um) apart.
    Expected outcome: the Z-separated nuclei yield 2 distinct labels while the
    closely spaced XY pair merges into 1 label.
    """
    ps = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    vol = _blob_volume((12, 40, 40), [(4, 20, 20), (7, 20, 20)], sigma=(0.4, 3.0, 3.0))
    vol[7] *= 0.9
    labels = voronoi_otsu_labeling(vol, spot_sigma=0.5, outline_sigma=0.5, pixel_sizes=ps)
    assert labels[4, 20, 20] > 0 and labels[7, 20, 20] > 0
    assert labels[4, 20, 20] != labels[7, 20, 20]

    vol_xy = _blob_volume((12, 40, 40), [(6, 20, 20), (6, 20, 22)], sigma=(0.4, 3.0, 3.0))
    labels_xy = voronoi_otsu_labeling(vol_xy, spot_sigma=0.5, outline_sigma=0.5, pixel_sizes=ps)
    assert len(np.unique(labels_xy[labels_xy > 0])) == 1


def test_voronoi_isotropic_matches_peak_local_max_reference():
    """Verify isotropic `voronoi_otsu_labeling` matches a direct scikit-image reference pipeline.

    Strategy: run `voronoi_otsu_labeling` without pixel sizes on a seeded random
    blurred 3D volume and compare against `peak_local_max(min_distance=3)` + watershed.
    Expected outcome: the labeled arrays are bit-for-bit identical.
    """
    rng = np.random.default_rng(0)
    vol = gaussian_filter(rng.random((24, 48, 48)), 2.0)
    labels = voronoi_otsu_labeling(vol, spot_sigma=1, outline_sigma=1)

    blurred = filters.gaussian(vol, sigma=(1, 1, 1))
    peaks = np.zeros_like(blurred, dtype=bool)
    peaks[tuple(feature.peak_local_max(blurred, min_distance=3, exclude_border=False).T)] = True
    outline = filters.gaussian(vol, sigma=(1, 1, 1))
    mask = outline > filters.threshold_otsu(outline)
    markers, _ = ndi.label(peaks)
    expected = segmentation.watershed(-outline, markers, mask=mask, compactness=0.5)
    np.testing.assert_array_equal(labels, expected)


def test_windowed_slice_projection_max():
    """Verify Z-axis max projection spreads a single impulse across the exact window span.

    Strategy: place an impulse of 10 at `(2, 5, 5)` in a `(5, 10, 10)` volume
    and project with `window_size=3, axis=0, operation='max'`.
    Expected outcome: slices 1, 2, and 3 at `(5, 5)` equal 10; slices 0 and 4 remain 0.
    """
    img = np.zeros((5, 10, 10))
    img[2, 5, 5] = 10

    thick = windowed_slice_projection(
        img, window_size=3, axis=0, operation="max", pixel_sizes=ISO_3D
    )
    assert thick.shape == (5, 10, 10)
    np.testing.assert_array_equal(thick[:, 5, 5], [0, 10, 10, 10, 0])


def test_windowed_slice_projection_physical_scaling_axes_and_window_rounding():
    """Verify physical window scaling (dict & tuple), `axis=1, 2`, even rounding, and `< 1` clamping.

    Strategy: project a single-voxel delta impulse at `(7, 7, 7)` in a `(15, 15, 15)`
    volume with anisotropic pixel sizes `{'Z': 2.0, 'Y': 0.5, 'X': 1.0}` across
    axes 0, 1, and 2, plus an even window (`4 -> 5`) and a sub-voxel window (`< 1 -> 1`).
    Expected outcome: impulse spreads over the exact odd pixel window on each
    axis for both `'max'` (value 1) and `'average'` (value `1 / win_px`), tuple
    and dict pixel sizes match identically, and sub-voxel windows preserve the impulse.
    """
    img = np.zeros((15, 15, 15), dtype=float)
    img[7, 7, 7] = 1.0
    ps_dict = {"Z": 2.0, "Y": 0.5, "X": 1.0}
    ps_tuple = (2.0, 0.5, 1.0)

    # axis=0 (Z=2.0): window_size=6.0 um -> 3 px (indices 6, 7, 8)
    max_z_dict = windowed_slice_projection(img, window_size=6.0, axis=0, operation="max", pixel_sizes=ps_dict)
    max_z_tuple = windowed_slice_projection(img, window_size=6.0, axis=0, operation="max", pixel_sizes=ps_tuple)
    np.testing.assert_array_equal(max_z_dict, max_z_tuple)
    assert np.array_equal(np.nonzero(max_z_dict[:, 7, 7])[0], [6, 7, 8])
    assert max_z_dict.sum() == 3.0

    avg_z = windowed_slice_projection(img, window_size=6.0, axis=0, operation="average", pixel_sizes=ps_dict)
    np.testing.assert_allclose(avg_z[6:9, 7, 7], 1.0 / 3.0)
    assert avg_z[5, 7, 7] == 0.0 and avg_z[9, 7, 7] == 0.0

    # axis=1 (Y=0.5): window_size=2.5 um -> 5 px (indices 5..9)
    max_y = windowed_slice_projection(img, window_size=2.5, axis=1, operation="max", pixel_sizes=ps_dict)
    assert np.array_equal(np.nonzero(max_y[7, :, 7])[0], [5, 6, 7, 8, 9])
    assert max_y.sum() == 5.0

    avg_y = windowed_slice_projection(img, window_size=2.5, axis=1, operation="average", pixel_sizes=ps_dict)
    np.testing.assert_allclose(avg_y[7, 5:10, 7], 1.0 / 5.0)

    # axis=2 (X=1.0): window_size=5.0 um -> 5 px (indices 5..9)
    max_x = windowed_slice_projection(img, window_size=5.0, axis=2, operation="max", pixel_sizes=ps_dict)
    assert np.array_equal(np.nonzero(max_x[7, 7, :])[0], [5, 6, 7, 8, 9])
    assert max_x.sum() == 5.0

    # Even window (4 -> 5 px)
    avg_even = windowed_slice_projection(img, window_size=4.0, axis=0, operation="average", pixel_sizes=ISO_3D)
    assert np.array_equal(np.nonzero(avg_even[:, 7, 7])[0], [5, 6, 7, 8, 9])
    np.testing.assert_allclose(avg_even[5:10, 7, 7], 1.0 / 5.0)

    # Sub-voxel window (0.2 / 2.0 = 0.1 -> clamped to 1 px: exact identity)
    for op in ("max", "average"):
        out_tiny = windowed_slice_projection(img, window_size=0.2, axis=0, operation=op, pixel_sizes=ps_dict)
        np.testing.assert_array_equal(out_tiny, img)


def test_windowed_slice_projection_monotonicity():
    """Verify max projection is monotonically non-decreasing with window size.

    Strategy: compute `'max'` projections with `window_size=3` and `window_size=5`
    on a seeded random 3D volume.
    Expected outcome: `proj_win5 >= proj_win3` holds at every voxel.
    """
    rng = np.random.default_rng(42)
    vol = rng.random((15, 10, 10)) * 100

    proj_win3 = windowed_slice_projection(vol, window_size=3, axis=0, operation="max", pixel_sizes=ISO_3D)
    proj_win5 = windowed_slice_projection(vol, window_size=5, axis=0, operation="max", pixel_sizes=ISO_3D)
    assert np.all(proj_win5 >= proj_win3)


def test_windowed_slice_projection_negative_values_and_unknown_operation():
    """Verify `'max'` handles negative values without zero-pad domination and rejects unknown ops.

    Strategy: project a strictly negative Z ramp with `'max'` and pass invalid
    `operation='mean'`.
    Expected outcome: negative values are preserved accurately across edges and
    unknown operations raise `ValueError`.
    """
    prof = np.array([-1, -2, -3, -4, -5], dtype=float)[:, None, None] * np.ones((1, 2, 2))
    out = windowed_slice_projection(prof, window_size=3, operation="max", pixel_sizes=ISO_3D)
    np.testing.assert_array_equal(out[:, 0, 0], [-1, -1, -2, -3, -4])
    with pytest.raises(ValueError):
        windowed_slice_projection(prof, window_size=3, operation="mean", pixel_sizes=ISO_3D)


def test_windowed_slice_projection_average_does_not_dim_edges():
    """Verify `'average'` normalizes by in-bounds slice count so edges are not dimmed by padding.

    Strategy: average a constant stack of ones with `window_size=11` and a
    linear Z ramp with `window_size=5`.
    Expected outcome: constant stack stays identically 1.0 everywhere (including
    boundary slices), interior ramp slices match the input, and slice 0 equals
    `mean([0, 1, 2])`.
    """
    stack = np.ones((12, 4, 4), dtype=np.float32)
    out = windowed_slice_projection(stack, window_size=11, operation="average", pixel_sizes=ISO_3D)
    np.testing.assert_allclose(out, 1.0)

    ramp = np.arange(30, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    out = windowed_slice_projection(ramp, window_size=5, operation="average", pixel_sizes=ISO_3D)
    np.testing.assert_allclose(out[2:-2, 0, 0], ramp[2:-2, 0, 0])
    np.testing.assert_allclose(out[0, 0, 0], np.mean([0, 1, 2]))


def test_optimized_entire_labels_touching_mask():
    """Verify full extent of touching labels is kept while non-touching labels are removed.

    Strategy: place two 2D label blocks and a mask within default expansion
    distance of label 1 only.
    Expected outcome: label 1 is retained completely and label 2 is zeroed out.
    """
    labels = np.zeros((20, 20), dtype=int)
    labels[5:10, 5:10] = 1
    labels[15:20, 15:20] = 2

    mask = np.zeros((20, 20), dtype=int)
    mask[2:4, 2:4] = 1

    res = optimized_entire_labels_touching_mask(labels, mask, pixel_sizes=ISO_2D)
    assert np.all(res[5:10, 5:10] == 1)
    assert np.all(res[15:20, 15:20] == 0)


def test_optimized_labels_touching_mask_float_distance_not_truncated():
    """Verify float `distance` is not truncated when `pixel_sizes=None` and matches unit pixel sizes.

    Strategy: place a mask `sqrt(5) ≈ 2.24` px from a label and expand by
    `distance=2.9` with `pixel_sizes=None` (warning expected) and with unit pixel sizes.
    Expected outcome: both return the same non-zero label (proving `2.9` was not truncated to `2`).
    """
    labels = np.zeros((1, 9, 9), dtype=int)
    labels[0, 4, 4] = 1
    mask = np.zeros_like(labels)
    mask[0, 6, 5] = 1

    with pytest.warns(UserWarning):
        implicit = optimized_entire_labels_touching_mask(labels, mask, distance=2.9)
    explicit = optimized_entire_labels_touching_mask(labels, mask, distance=2.9, pixel_sizes=ISO_3D)
    np.testing.assert_array_equal(implicit, explicit)
    assert implicit[0, 4, 4] == 1


def test_optimized_entire_labels_touching_mask_anisotropic_distance():
    """Verify expansion distance is evaluated in physical units on anisotropic grids.

    Strategy: compare a label 3 slices away in Z (`9 um` at `Z=3 um`) against a
    label 3 px away in X (`3 um` at `X=1 um`) using `distance=5.0 um`, plus an
    isotropic negative control.
    Expected outcome: anisotropic run retains only the X-neighbouring label;
    isotropic control retains the Z-neighbouring label too.
    """
    aniso = {"Z": 3.0, "Y": 1.0, "X": 1.0}
    mask = np.zeros((9, 9, 9), dtype=int)
    mask[4, 4, 4] = 1

    below = np.zeros_like(mask)
    below[1, 4, 4] = 1  # 3 slices from mask: 9 um
    beside = np.zeros_like(mask)
    beside[4, 4, 1] = 1  # 3 pixels from mask: 3 um

    assert optimized_entire_labels_touching_mask(below, mask, distance=5.0, pixel_sizes=aniso).max() == 0
    assert optimized_entire_labels_touching_mask(beside, mask, distance=5.0, pixel_sizes=aniso).max() == 1
    assert optimized_entire_labels_touching_mask(below, mask, distance=5.0, pixel_sizes=ISO_3D).max() == 1


def test_sample_intensity_around_points():
    """Verify 3D mean sampling around interior points and NaN for positive and negative OOB coordinates.

    Strategy: sample around an impulse voxel, a uniform region, and both
    positive and negative out-of-bounds coordinates with `diameter=2.0` (3x3x3 box).
    Expected outcome: center point returns `(26 + 10) / 27`, uniform point
    returns `1.0`, and all OOB coordinates (positive and negative) return `NaN`.
    """
    image_3d = np.ones((10, 10, 10))
    image_3d[5, 5, 5] = 10

    points = np.array(
        [
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
            [20.0, 20.0, 20.0],
            [-1.0, 5.0, 5.0],
            [5.0, -2.0, 5.0],
            [5.0, 5.0, -0.6],
        ]
    )

    res = sample_intensity_around_points(image_3d, points, diameter=2.0, pixel_sizes=ISO_3D)
    assert len(res) == 6
    assert np.isclose(res[0], 36 / 27)
    assert np.isclose(res[1], 1.0)
    assert all(np.isnan(res[i]) for i in range(2, 6))


def test_sample_intensity_around_points_anisotropic_impulse_footprint():
    """Verify anisotropic pixel sizes produce the exact expected box footprint in 3D and 2D.

    Strategy: place a unit impulse at the center of a 3D `(15, 15, 15)` volume
    with `pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 0.5}` and `diameter=5.0`
    (expected box `3 x 5 x 11 = 165` voxels), plus a 2D `(15, 15)` image with
    `{'Y': 1.0, 'X': 0.5}` and `diameter=3.0` (`3 x 7 = 21` pixels).
    Expected outcome: points inside the anisotropic footprint evaluate to exact
    `1 / volume`, points one voxel outside evaluate to `0.0`, and tuple pixel
    sizes match dict pixel sizes identically.
    """
    img = np.zeros((15, 15, 15), dtype=float)
    img[7, 7, 7] = 1.0
    ps_dict = {"Z": 2.0, "Y": 1.0, "X": 0.5}
    ps_tuple = (2.0, 1.0, 0.5)

    # Footprints: Z=3 (6..8), Y=5 (5..9), X=11 (2..12) -> 165 voxels
    pts = [
        [7, 7, 7],  # center -> 1/165
        [6, 7, 7],  # Z edge inside -> 1/165
        [5, 7, 7],  # Z outside -> 0
        [7, 5, 7],  # Y edge inside -> 1/165
        [7, 4, 7],  # Y outside -> 0
        [7, 7, 2],  # X edge inside -> 1/165
        [7, 7, 1],  # X outside -> 0
    ]
    expected = [1.0 / 165, 1.0 / 165, 0.0, 1.0 / 165, 0.0, 1.0 / 165, 0.0]

    res_dict = sample_intensity_around_points(img, pts, diameter=5.0, pixel_sizes=ps_dict)
    res_tuple = sample_intensity_around_points(img, pts, diameter=5.0, pixel_sizes=ps_tuple)
    np.testing.assert_allclose(res_dict, expected, atol=1e-14)
    np.testing.assert_allclose(res_tuple, expected, atol=1e-14)

    # 2D input: Y=3 (6..8), X=7 (4..10) -> 21 pixels
    img_2d = np.zeros((15, 15), dtype=float)
    img_2d[7, 7] = 1.0
    ps_2d = {"Y": 1.0, "X": 0.5}
    res_2d = sample_intensity_around_points(
        img_2d, [[7, 7], [7, 4], [7, 3], [-1, 7], [15, 7]], diameter=3.0, pixel_sizes=ps_2d
    )
    np.testing.assert_allclose(res_2d[:3], [1.0 / 21, 1.0 / 21, 0.0], atol=1e-14)
    assert np.isnan(res_2d[3]) and np.isnan(res_2d[4])
    assert sample_intensity_around_points(img_2d, [7, 7], diameter=3.0, pixel_sizes=ps_2d) == [
        pytest.approx(1.0 / 21)
    ]


def test_sample_intensity_around_points_constant_conservation():
    """Verify constant intensity is conserved everywhere (interior and boundary via `mode='reflect'`).

    Strategy: sample corners, face boundaries, and 30 seeded random points in a
    uniform `np.pi` volume using `diameter=5.0`.
    Expected outcome: every sample returns `np.pi` within `1e-14` with no boundary dilution.
    """
    img = np.full((12, 12, 12), np.pi)
    rng = np.random.default_rng(42)
    boundary_pts = np.array([[0, 0, 0], [0, 5, 5], [11, 11, 11], [11, 0, 5]], dtype=float)
    random_pts = rng.uniform(0.0, 11.0, size=(30, 3))
    points = np.vstack([boundary_pts, random_pts])

    sampled = sample_intensity_around_points(img, points, diameter=5.0, pixel_sizes=ISO_3D)
    np.testing.assert_allclose(sampled, np.pi, atol=1e-14)


def test_sample_intensity_linear_scaling():
    """Verify local mean sampling obeys scalar linearity `f(c * img) = c * f(img)`.

    Strategy: sample a seeded random 3D volume and its `10x` scaled copy at the
    same 10 coordinates.
    Expected outcome: scaled outputs equal `10 * base` within `1e-12`.
    """
    rng = np.random.default_rng(43)
    img = rng.random((20, 20, 20)) * 100.0
    points = rng.uniform(0.0, 19.0, size=(10, 3))

    sampled_base = np.array(sample_intensity_around_points(img, points, diameter=3, pixel_sizes=ISO_3D))
    sampled_scaled = np.array(
        sample_intensity_around_points(img * 10.0, points, diameter=3, pixel_sizes=ISO_3D)
    )
    np.testing.assert_allclose(sampled_scaled, sampled_base * 10.0, atol=1e-12)


def test_sample_intensity_xyz_warning():
    """Verify likely `(X, Y, Z)` point order triggers a `UserWarning` and yields `NaN` when OOB.

    Strategy: pass `[25, 10, 2]` to a `(5, 20, 30)` volume with explicit pixel sizes.
    Expected outcome: emits the XYZ-order `UserWarning` and returns `NaN`.
    """
    image_3d = np.zeros((5, 20, 30))
    points_xyz = np.array([[25, 10, 2]])

    with pytest.warns(UserWarning, match=r"Points appear to be in \(X, Y, Z\) order"):
        res = sample_intensity_around_points(image_3d, points_xyz, diameter=3, pixel_sizes=ISO_3D)
    assert np.isnan(res[0])


def test_sample_intensity_around_points_integer_means_not_truncated():
    """Verify integer images are averaged in float64 rather than truncated to integer dtype.

    Strategy: compare sampled means on a `uint16` Poisson(5) volume against its
    `float64` cast, plus a single `255` voxel in a `uint8` 3-cube.
    Expected outcome: `uint16` matches `float64` within `1e-9` and the `uint8`
    impulse evaluates to `255 / 27`.
    """
    rng = np.random.default_rng(1)
    img = rng.poisson(5.0, size=(40, 40, 40)).astype(np.uint16)
    pts = rng.integers(5, 35, size=(200, 3))
    means_int = np.array(sample_intensity_around_points(img, pts, diameter=5, pixel_sizes=ISO_3D))
    means_float = np.array(
        sample_intensity_around_points(img.astype(np.float64), pts, diameter=5, pixel_sizes=ISO_3D)
    )
    np.testing.assert_allclose(means_int, means_float, rtol=0, atol=1e-9)
    assert abs(means_int.mean() - 5.0) < 0.15
    assert len(np.unique(means_int)) > 10

    one = np.zeros((9, 9, 9), dtype=np.uint8)
    one[4, 4, 4] = 255
    val = sample_intensity_around_points(one, [[4, 4, 4]], diameter=3, pixel_sizes=ISO_3D)[0]
    assert val == pytest.approx(255 / 27)


def test_sample_intensity_around_points_window_matches_diameter():
    """Verify odd window side length matches diameter without banker's rounding drift.

    Strategy: sample a unit impulse at `(7, 7, 7)` for diameters 1, 2, 3, 4, 5, 7.
    Expected outcome: each diameter yields `1 / side^3` with sides 1, 3, 3, 5, 5, 7.
    """
    one = np.zeros((15, 15, 15))
    one[7, 7, 7] = 1.0
    for diameter, side in [(1, 1), (2, 3), (3, 3), (4, 5), (5, 5), (7, 7)]:
        val = sample_intensity_around_points(one, [[7, 7, 7]], diameter=diameter, pixel_sizes=ISO_3D)[0]
        assert val == pytest.approx(1.0 / side**3), diameter


def test_sample_intensity_around_points_empty_and_flat_points():
    """Verify empty point list, single flat point vector, and mismatched dimensionality handling.

    Strategy: pass `[]`, `[2, 2, 2]`, and a `(4, 2)` array for a 3D volume.
    Expected outcome: returns `[]` and `[1.0]` respectively, and raises `ValueError` for `(4, 2)`.
    """
    img = np.ones((5, 5, 5))
    assert sample_intensity_around_points(img, [], 3, ISO_3D) == []
    assert sample_intensity_around_points(img, [2, 2, 2], 3, ISO_3D) == [1.0]
    with pytest.raises(ValueError):
        sample_intensity_around_points(img, np.zeros((4, 2)), 3, ISO_3D)


def test_sample_intensity_along_surface_normals_anisotropic_profile():
    """Verify anisotropic normal sampling recovers a symmetric Gaussian Z profile centered on the surface.

    Strategy: place a planar grid at `Z=5` (`10 um` at `Z=2 um`) inside a
    Gaussian-in-Z volume peaked at `10 um` and sample with `thickness=8 um, num_steps=5`.
    Expected outcome: output shape is `(6, 6, 5)`, peak lies at step 2 (offset 0),
    and `+/-` offset intensities are symmetric.
    """
    pixel_sizes = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    z_phys = np.arange(11) * pixel_sizes["Z"]
    gaussian_z = np.exp(-((z_phys - 10.0) ** 2) / (2 * (2.0**2))) * 100.0
    img = np.broadcast_to(gaussian_z[:, None, None], (11, 20, 20)).copy()

    u_grid, v_grid = np.meshgrid(np.linspace(5, 15, 6), np.linspace(5, 15, 6), indexing="ij")
    surface_grid = np.stack([np.full_like(u_grid, 5.0), u_grid, v_grid], axis=-1)

    sampled = sample_intensity_along_surface_normals(
        img, surface_grid, thickness=8.0, num_steps=5, interpolation="cubic", pixel_sizes=pixel_sizes
    )
    assert sampled.shape == (6, 6, 5)
    assert np.all(np.argmax(sampled, axis=-1) == 2)

    profile = sampled[3, 3, :]
    assert np.isclose(profile[0], profile[4])
    assert np.isclose(profile[1], profile[3])
    assert profile[2] > profile[1] > profile[0]


def test_sample_intensity_along_surface_normals_curved_surface_bicubic():
    """Verify bicubic normal sampling over a curved spherical shell and default `pixel_sizes=None` warning.

    Strategy: construct a spherical Gaussian shell at `r=8` and sample normals
    from a matching spherical mesh with `pixel_sizes=None`.
    Expected outcome: emits `UserWarning`, produces finite `(8, 12, 7)` output,
    and peaks at the middle step (index 3) across interior latitudes.
    """
    z, y, x = np.ogrid[:31, :31, :31]
    radius_grid = np.sqrt((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2)
    img = np.exp(-((radius_grid - 8.0) ** 2) / (2 * (2.0**2))) * 250.0

    theta = np.linspace(0, np.pi, 8)
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    r = 8.0
    surface_grid = np.stack(
        [
            15.0 + r * np.cos(theta_grid),
            15.0 + r * np.sin(theta_grid) * np.sin(phi_grid),
            15.0 + r * np.sin(theta_grid) * np.cos(phi_grid),
        ],
        axis=-1,
    )

    with pytest.warns(UserWarning, match="pixel_sizes not provided"):
        sampled = sample_intensity_along_surface_normals(
            img, surface_grid, thickness=4.0, num_steps=7, interpolation="cubic", pixel_sizes=None
        )
    assert sampled.shape == (8, 12, 7)
    assert not np.isnan(sampled).any()
    assert np.all(np.argmax(sampled[2:6, :, :], axis=-1) == 3)


def test_surface_normals_single_step_samples_surface():
    """Verify `num_steps=1` samples directly on the surface (offset 0) rather than at `-thickness/2`.

    Strategy: sample a linear Z ramp at `Z=10` with `thickness=4, num_steps=1`.
    Expected outcome: all sampled values equal `10.0`.
    """
    vol = np.broadcast_to(np.arange(20, dtype=float)[:, None, None], (20, 20, 20)).copy()
    yy, xx = np.meshgrid(np.arange(5, 15), np.arange(5, 15), indexing="ij")
    grid = np.stack([np.full_like(yy, 10.0), yy, xx], axis=-1).astype(float)
    out = sample_intensity_along_surface_normals(
        vol, grid, thickness=4, num_steps=1, pixel_sizes=ISO_3D
    )
    assert np.all(out == 10.0)


def test_surface_normals_interpolation_degenerate_and_oob():
    """Verify nearest vs bicubic subpixel behavior, degenerate zero normals, and OOB `cval=0`.

    Strategy: (1) sample a linear Z ramp at subpixel offsets `+/- 0.3` with
    `'nearest'` vs `'bicubic'`, (2) pass a collapsed constant mesh where
    `t_u x t_v == 0`, and (3) sample a grid sitting on `Z=0` so negative offsets
    leave the volume.
    Expected outcome: `'nearest'` rounds to the center slice (`100`) while
    `'bicubic'` recovers `[97, 100, 103]`; degenerate normals yield zero offset
    without NaNs; out-of-bounds normal steps return `0.0`.
    """
    ramp = np.broadcast_to(np.arange(20, dtype=float)[:, None, None] * 10.0, (20, 10, 10)).copy()
    yy, xx = np.meshgrid(np.arange(3, 7), np.arange(3, 7), indexing="ij")
    grid = np.stack([np.full_like(yy, 10.0), yy, xx], axis=-1).astype(float)

    near = sample_intensity_along_surface_normals(
        ramp, grid, thickness=0.6, num_steps=3, interpolation="nearest", pixel_sizes=ISO_3D
    )
    cubic = sample_intensity_along_surface_normals(
        ramp, grid, thickness=0.6, num_steps=3, interpolation="cubic", pixel_sizes=ISO_3D
    )
    np.testing.assert_allclose(near, 100.0)
    np.testing.assert_allclose(cubic[1, 1, :], [97.0, 100.0, 103.0], atol=1e-2)

    # Degenerate surface (constant coordinates -> zero normal) must not produce NaN
    collapsed = np.full((3, 3, 3), 5.0)
    deg_out = sample_intensity_along_surface_normals(
        ramp, collapsed, thickness=4.0, num_steps=3, interpolation="nearest", pixel_sizes=ISO_3D
    )
    assert not np.isnan(deg_out).any()
    np.testing.assert_allclose(deg_out, ramp[5, 5, 5])

    # Out-of-bounds normal steps fill with cval=0
    const_vol = np.full((10, 10, 10), 42.0)
    top_grid = np.stack([np.zeros_like(yy), yy, xx], axis=-1).astype(float)
    oob_out = sample_intensity_along_surface_normals(
        const_vol, top_grid, thickness=4.0, num_steps=5, interpolation="nearest", pixel_sizes=ISO_3D
    )
    np.testing.assert_allclose(oob_out[1, 1, :], [0.0, 0.0, 42.0, 42.0, 42.0])


def test_pixel_sizes_parsing_and_validation_via_public_apis():
    """Verify dict, partial dict, sequence, default `None`, and invalid pixel sizes via public APIs.

    Strategy: compare `create_ellipsoid_struct` outputs for dict, partial dict
    (missing `'Y'` defaults to `1.0`), list/tuple sequences, and `None` (warning),
    then verify invalid values (zero, negative, NaN, +/-inf, wrong length) raise
    `ValueError` through `create_ellipsoid_struct` and `sample_intensity_around_points`.
    Expected outcome: equivalent pixel-size representations yield identical
    structuring elements and all invalid specifications raise `ValueError`.
    """
    from_dict = create_ellipsoid_struct(2.0, pixel_sizes={"Z": 2.0, "Y": 1.0, "X": 0.5})
    from_partial = create_ellipsoid_struct(2.0, pixel_sizes={"Z": 2.0, "X": 0.5})
    from_list = create_ellipsoid_struct(2.0, pixel_sizes=[2.0, 1.0, 0.5])
    np.testing.assert_array_equal(from_dict, from_partial)
    np.testing.assert_array_equal(from_dict, from_list)

    with pytest.warns(UserWarning, match="pixel_sizes not provided"):
        from_none = create_ellipsoid_struct(2.0)
    np.testing.assert_array_equal(from_none, create_ellipsoid_struct(2.0, pixel_sizes=ISO_3D))

    invalid_cases = [
        {"Z": 0.0, "Y": 1.0, "X": 1.0},
        {"Z": -1.0, "Y": 1.0, "X": 1.0},
        (-0.5, 1.0, 1.0),
        {"Z": np.nan, "Y": 1.0, "X": 1.0},
        {"Z": np.inf, "Y": 1.0, "X": 1.0},
        (1.0, -np.inf, 1.0),
        (1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    vol = np.ones((4, 4, 4))
    for bad in invalid_cases:
        with pytest.raises(ValueError):
            create_ellipsoid_struct(2.0, pixel_sizes=bad)
        with pytest.raises(ValueError):
            sample_intensity_around_points(vol, [[2, 2, 2]], diameter=3, pixel_sizes=bad)


def test_none_pixel_size_falls_back_to_one_with_warning():
    """Verify `None` axis entries from uncalibrated readers warn and fall back to 1.0.

    Strategy: pass `{'Z': None, 'Y': 1.0, 'X': 1.0}` to
    `optimized_entire_labels_touching_mask` and `sample_intensity_along_surface_normals`.
    Expected outcome: each call warns naming `axis Z` and produces valid outputs
    equivalent to `Z=1.0`.
    """
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[1:5, 1:5, 1:5] = 1
    labels[6:10, 6:10, 6:10] = 2
    mask = np.zeros_like(labels)
    mask[2:4, 2:4, 2:4] = 1
    with pytest.warns(UserWarning, match="axis Z"):
        out = optimized_entire_labels_touching_mask(
            labels, mask, distance=1, pixel_sizes={"Z": None, "Y": 1.0, "X": 1.0}
        )
    np.testing.assert_array_equal(out, labels * (labels == 1))

    vol = np.full((20, 20, 20), 100.0)
    yy, xx = np.meshgrid(np.arange(5, 15), np.arange(5, 15), indexing="ij")
    grid = np.stack([np.full_like(yy, 10.0), yy, xx], axis=-1).astype(float)
    with pytest.warns(UserWarning, match="axis Z"):
        sampled = sample_intensity_along_surface_normals(
            vol, grid, pixel_sizes={"X": 0.5, "Y": 0.5, "Z": None}
        )
    assert np.all(sampled == 100.0)


def test_pixel_size_units_are_converted_to_um_and_missing_z_uses_finest_xy():
    """Verify processing functions read pixel sizes in µm whatever the dict's unit (C-02, D01).

    Strategy: (+) build ``create_ellipsoid_struct(1.0)`` (radius 1 µm) with
    the same 0.1 µm pixels given in nm, mm-free µm and as a plain tuple;
    (+) a dict with Z missing and Y/X known; (-) a non-length unit warns.
    Expected outcome: identical structuring elements (21 px across at 0.1 µm),
    not the 3x3x3 element nm values used to give; the missing Z takes 0.1 µm
    with a warning naming axis Z; a non-length unit ('a.u.') warns and is used
    as µm; the unit 'pixel' means uncalibrated (all axes unknown, 1.0, warning).
    """
    ref = create_ellipsoid_struct(1.0, pixel_sizes=(0.1, 0.1, 0.1))
    assert ref.shape == (21, 21, 21)
    nm = create_ellipsoid_struct(1.0, pixel_sizes={"Z": 100.0, "Y": 100.0, "X": 100.0, "unit": "nm"})
    np.testing.assert_array_equal(nm, ref)
    with pytest.warns(UserWarning, match="axis Z"):
        no_z = create_ellipsoid_struct(1.0, pixel_sizes={"y": 0.1, "x": 0.1, "unit": "µm"})
    np.testing.assert_array_equal(no_z, ref)
    with pytest.warns(UserWarning, match="not a length unit"):
        au = create_ellipsoid_struct(1.0, pixel_sizes={"Z": 0.1, "Y": 0.1, "X": 0.1, "unit": "a.u."})
    np.testing.assert_array_equal(au, ref)
    # 'pixel' (ImageJ's uncalibrated unit) is no calibration: every axis unknown -> 1.0, with a warning
    with pytest.warns(UserWarning, match="no value for axis Z, Y, X"):
        px = create_ellipsoid_struct(1.0, pixel_sizes={"Z": 0.1, "Y": 0.1, "X": 0.1, "unit": "pixel"})
    assert px.shape == (3, 3, 3)


@pytest.mark.parametrize(
    "normal",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, -2.0, 5.0]),
    ],
)
def test_generate_plane_basis(normal):
    """Verify `generate_plane_basis` builds a right-handed orthonormal triad for cardinal and oblique normals.

    Strategy: generate `(u, v)` across all three cardinal axes, a scaled negative
    cardinal normal, and oblique normals.
    Expected outcome: `u` and `v` are unit length, mutually orthogonal,
    orthogonal to `normal`, and satisfy `u x v == normal / ||normal||`.
    """
    n_hat = normal / np.linalg.norm(normal)
    u, v = generate_plane_basis(normal)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.dot(u, n_hat), 0.0, atol=1e-12)
    assert np.isclose(np.dot(v, n_hat), 0.0, atol=1e-12)
    assert np.isclose(np.dot(u, v), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(u, v), n_hat, atol=1e-12)


def test_generate_plane_basis_zero_normal_raises():
    """Verify `generate_plane_basis` rejects the zero vector with `ValueError`.

    Strategy: call `generate_plane_basis(np.zeros(3))`.
    Expected outcome: raises `ValueError` mentioning non-zero normal.
    """
    with pytest.raises(ValueError, match="non-zero"):
        generate_plane_basis(np.zeros(3))


def test_fit_plane_ransac():
    """Verify RANSAC plane fitting on isotropic horizontal and anisotropic tilted planes plus error cases.

    Strategy: fit (1) a horizontal plane with an outlier, (2) an anisotropic
    tilted physical plane `z_phys = 0.5 * y_phys + 0.25 * x_phys + 4.0` with
    3 gross outliers, (3) fewer than 3 points, and (4) collinear points.
    Expected outcome: recovers exact physical normals with canonical sign
    `normal[0] >= 0` and inlier centroid on the plane; `< 3` points raises
    `ValueError` and collinear points raise `RuntimeError`.
    """
    points_zyx = np.array(
        [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
    )
    p0, normal = fit_plane_ransac(points_zyx, pixel_sizes=ISO_3D, inlier_threshold_um=0.1)
    assert np.isclose(p0[0], 2.0)
    assert normal[0] >= 0.0
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-6)

    # Anisotropic tilted plane: z_phys - 0.5*y_phys - 0.25*x_phys = 4.0
    ps = {"Z": 2.0, "Y": 1.0, "X": 0.5}
    yy, xx = np.meshgrid(np.linspace(0.0, 8.0, 5), np.linspace(0.0, 8.0, 5), indexing="ij")
    y_phys = yy.ravel()
    x_phys = xx.ravel()
    z_phys = 0.5 * y_phys + 0.25 * x_phys + 4.0
    inliers_vox = np.column_stack([z_phys / ps["Z"], y_phys / ps["Y"], x_phys / ps["X"]])
    outliers_vox = np.array([[50.0, 2.0, 4.0], [60.0, 5.0, 1.0], [-40.0, 3.0, 6.0]])
    pts_all = np.vstack([inliers_vox, outliers_vox])

    p0_t, n_t = fit_plane_ransac(pts_all, pixel_sizes=ps, inlier_threshold_um=0.05)
    expected_n = np.array([1.0, -0.5, -0.25]) / np.linalg.norm([1.0, -0.5, -0.25])
    assert n_t[0] >= 0.0
    np.testing.assert_allclose(n_t, expected_n, atol=1e-6)
    assert np.isclose(np.dot(p0_t, n_t), 4.0 / np.linalg.norm([1.0, -0.5, -0.25]), atol=1e-6)

    with pytest.raises(ValueError, match="At least 3 points"):
        fit_plane_ransac(np.zeros((2, 3)), pixel_sizes=ISO_3D)

    collinear = np.column_stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)])
    with pytest.raises(RuntimeError, match="RANSAC failed"):
        fit_plane_ransac(collinear, pixel_sizes=ISO_3D, max_iters=50)


def test_sample_volume_plane_normalises_normal_and_labels_axes_zyx():
    """Verify `sample_volume_plane` normalizes `normal_phys` and returns ZYX output pixel sizes.

    Strategy: sample a Z-ramp volume with `normal_phys=[3, 0, 0]` and `[1, 0, 0]`
    at `num_offsets=1, offset_step_um=1.0`.
    Expected outcome: both runs agree identically, offset slices land at Z = 9, 10, 11,
    output shape is `(3, 9, 5)`, and pixel sizes map `u -> Y` and `v -> X`.
    """
    vol = np.broadcast_to(np.arange(20, dtype=np.float32)[:, None, None], (20, 20, 20)).copy()
    kw = dict(
        pixel_sizes=ISO_3D,
        p0_phys=np.array([10.0, 10.0, 10.0]),
        u_range_um=(-4, 4),
        v_range_um=(-4, 4),
        u_res=9,
        v_res=5,
        num_offsets=1,
        offset_step_um=1.0,
    )
    a, sp_a = sample_volume_plane(vol, normal_phys=np.array([3.0, 0.0, 0.0]), **kw)
    b, _ = sample_volume_plane(vol, normal_phys=np.array([1.0, 0.0, 0.0]), **kw)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:, 4, 2], [9, 10, 11])
    assert a.shape == (3, 9, 5)
    assert sp_a == {"Z": 1.0, "Y": 1.0, "X": 2.0}


def test_sample_volume_plane_cubic_interpolates():
    """Verify prefiltered cubic sampling interpolates exact voxel centers and matches whole-volume reference.

    Strategy: sample a random `float32` volume on an axis-aligned voxel-center
    grid (comparing cubic vs linear) and on an oblique plane (comparing cropped
    prefilter against full-volume `map_coordinates(..., order=3, prefilter=True)`).
    Expected outcome: voxel-center samples match linear within `1e-5` and oblique
    cubic samples match the full-volume reference within `1e-4`.
    """
    rng = np.random.default_rng(3)
    vol = rng.random((12, 16, 16)).astype(np.float32)
    n = np.array([1.0, 0.0, 0.0])
    plane, _ = sample_volume_plane(
        vol,
        ISO_3D,
        p0_phys=np.array([5.0, 0.0, 0.0]),
        normal_phys=n,
        u_range_um=(-15, 15),
        v_range_um=(-15, 15),
        u_res=31,
        v_res=31,
        method="cubic",
    )
    linear, _ = sample_volume_plane(
        vol,
        ISO_3D,
        p0_phys=np.array([5.0, 0.0, 0.0]),
        normal_phys=n,
        u_range_um=(-15, 15),
        v_range_um=(-15, 15),
        u_res=31,
        v_res=31,
        method="linear",
    )
    np.testing.assert_allclose(plane, linear, atol=1e-5)
    assert np.count_nonzero(linear) == 16 * 16

    vol2 = rng.random((40, 40, 40)).astype(np.float32)
    n2 = np.array([1.0, 0.3, -0.2])
    plane2, _ = sample_volume_plane(
        vol2,
        ISO_3D,
        p0_phys=np.array([20.0, 20.0, 20.0]),
        normal_phys=n2,
        u_range_um=(-6, 6),
        v_range_um=(-6, 6),
        u_res=25,
        v_res=25,
        method="cubic",
    )
    u_dir, v_dir = generate_plane_basis(n2)
    uu, vv = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(-6, 6, 25), indexing="ij")
    pts = (
        np.array([20.0, 20.0, 20.0])[:, None, None]
        + uu * u_dir[:, None, None]
        + vv * v_dir[:, None, None]
    )
    expected = map_coordinates(
        vol2, pts.reshape(3, -1), order=3, mode="constant", cval=0, prefilter=True
    ).reshape(25, 25)
    np.testing.assert_allclose(plane2, expected, atol=1e-4)


def test_sample_volume_plane_dynamic_bounds_uint8_clamping_oob_and_errors():
    """Verify dynamic extent/resolution, `uint8` cubic overshoot clamping, outside-volume cubic, and errors.

    Strategy: (1) sample with `u_range_um=v_range_um=u_res=v_res=None` on an
    anisotropic constant volume, (2) sample a sharp `0 -> 255` `uint8` step at
    half-voxel coordinates using `method='cubic'`, (3) sample a cubic plane far
    outside the volume, and (4) test missing `p0_phys`/`normal_phys` and zero normal.
    Expected outcome: dynamic bounds pick exact finest spacing (`0.5`), `uint8`
    cubic overshoot clamps to `[0, 255]` without wraparound, out-of-volume cubic
    returns all zeros, and invalid plane definitions raise `ValueError`.
    """
    # 1. Dynamic bounds & resolution: Z=5*2=10, Y=7*1=7, X=9*0.5=4.5; optimal_res=0.5
    # For normal=[1,0,0], u is +X (span 4.5 -> 10 px), v is -Y (span 7.0 -> 15 px).
    vol = np.full((6, 8, 10), 7.0, dtype=np.float32)
    ps = {"Z": 2.0, "Y": 1.0, "X": 0.5}
    dyn_plane, dyn_ps = sample_volume_plane(
        vol, pixel_sizes=ps, p0_phys=np.array([4.0, 3.0, 2.0]), normal_phys=np.array([1.0, 0.0, 0.0])
    )
    assert dyn_plane.shape == (10, 15)
    np.testing.assert_allclose(dyn_plane, 7.0)
    # D01 (R8-23): a single plane has no Z step -> None, not 0.0, so the dict
    # can be passed back as pixel_sizes= (0.0 is rejected by every parser)
    assert dyn_ps == {"Z": None, "Y": pytest.approx(0.5), "X": pytest.approx(0.5)}
    with pytest.warns(UserWarning, match="axis Z"):  # Z takes the finest of Y/X
        assert create_ellipsoid_struct(1.0, pixel_sizes=dyn_ps).shape == (5, 5, 5)

    # 2. Integer dtype + cubic overshoot clamping (no uint8 underflow/overflow wraparound)
    step_u8 = np.zeros((10, 16, 16), dtype=np.uint8)
    step_u8[:, :, 8:] = 255
    # u is +X: u_vals in [-7, 7] step 0.5 -> X in [1.0, 15.0]; idx 11 is X=6.5, idx 15 is X=8.5
    clamped, _ = sample_volume_plane(
        step_u8,
        pixel_sizes=ISO_3D,
        p0_phys=np.array([5.0, 8.0, 8.0]),
        normal_phys=np.array([1.0, 0.0, 0.0]),
        u_range_um=(-7.0, 7.0),
        v_range_um=(-3.0, 3.0),
        u_res=29,
        v_res=7,
        method="cubic",
    )
    assert clamped.dtype == np.uint8
    assert np.all(clamped[11, :] == 0)
    assert np.all(clamped[15, :] == 255)
    assert np.all(clamped[:13, :] < 30)
    assert np.all(clamped[14:, :] > 225)

    # 3. Outside-volume cubic returns zeros cleanly
    oob_plane, _ = sample_volume_plane(
        np.ones((10, 10, 10), dtype=np.float32),
        pixel_sizes=ISO_3D,
        p0_phys=np.array([100.0, 100.0, 100.0]),
        normal_phys=np.array([1.0, 0.0, 0.0]),
        u_range_um=(-2.0, 2.0),
        v_range_um=(-2.0, 2.0),
        u_res=5,
        v_res=5,
        method="cubic",
    )
    np.testing.assert_array_equal(oob_plane, 0.0)

    # 4. Missing p0_phys / normal_phys or zero normal_phys raises ValueError
    with pytest.raises(ValueError, match="must be provided"):
        sample_volume_plane(vol, pixel_sizes=ps, p0_phys=None, normal_phys=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="must be provided"):
        sample_volume_plane(vol, pixel_sizes=ps, p0_phys=np.zeros(3), normal_phys=None)
    with pytest.raises(ValueError, match="non-zero"):
        sample_volume_plane(vol, pixel_sizes=ps, p0_phys=np.zeros(3), normal_phys=np.zeros(3))


def test_create_ellipsoid_struct():
    """Verify exact voxel membership, reflection symmetry, tiny radius, and default warning.

    Strategy: inspect exact on/off voxel coordinates and 3-axis symmetry of
    isotropic and anisotropic (`Z=2, Y=1, X=1`) ellipsoids of radius `3 um`,
    plus a tiny `1e-8 um` radius and a `None` spacing warning check.
    Expected outcome: shapes match `2 * ceil(r / s) + 1`, boundary voxels obey
    `sum((d * s)^2) <= r^2`, struct is symmetric along all 3 axes, and `None`
    spacing warns and defaults to `(1, 1, 1)`.
    """
    struct_iso = create_ellipsoid_struct(radius_um=3.0, pixel_sizes=ISO_3D)
    assert struct_iso.shape == (7, 7, 7) and struct_iso.dtype == bool
    assert struct_iso[3, 3, 3]
    assert struct_iso[0, 3, 3] and not struct_iso[0, 0, 0]

    # Anisotropic spacing (2, 1, 1), radius 3 um -> radii_vox (1.5, 3, 3) -> shape (5, 7, 7), center (2, 3, 3)
    struct_aniso = create_ellipsoid_struct(radius_um=3.0, pixel_sizes=(2.0, 1.0, 1.0))
    assert struct_aniso.shape == (5, 7, 7)
    assert struct_aniso[2, 3, 3]
    assert struct_aniso[1, 3, 3] and struct_aniso[3, 3, 3]  # |dz|=1 -> 2 um <= 3 um
    assert not struct_aniso[0, 3, 3] and not struct_aniso[4, 3, 3]  # |dz|=2 -> 4 um > 3 um
    assert struct_aniso[2, 0, 3] and struct_aniso[2, 6, 3]  # |dy|=3 -> 3 um <= 3 um
    assert struct_aniso[3, 5, 4]  # 2^2 + 2^2 + 1^2 = 9 <= 9
    assert not struct_aniso[3, 5, 5]  # 2^2 + 2^2 + 2^2 = 12 > 9

    # Reflection symmetry across all 3 axes
    np.testing.assert_array_equal(struct_aniso, struct_aniso[::-1, :, :])
    np.testing.assert_array_equal(struct_aniso, struct_aniso[:, ::-1, :])
    np.testing.assert_array_equal(struct_aniso, struct_aniso[:, :, ::-1])

    # Tiny radius preserves only the center voxel
    struct_tiny = create_ellipsoid_struct(radius_um=1e-8, pixel_sizes=(1.0, 1.0, 1.0))
    assert struct_tiny.shape == (3, 3, 3)
    assert struct_tiny[1, 1, 1] and np.sum(struct_tiny) == 1

    # Default None spacing warns and equals unit spacing
    with pytest.warns(UserWarning, match="pixel_sizes not provided"):
        struct_default = create_ellipsoid_struct(radius_um=2.0)
    np.testing.assert_array_equal(struct_default, create_ellipsoid_struct(radius_um=2.0, pixel_sizes=ISO_3D))


def test_generate_morphological_surface_mask():
    """Verify morphological surface extraction, `fill_holes=False`, thin-depth fallback, and empty input.

    Strategy: run on (1) a solid sphere, (2) a hollow cube with `fill_holes=True`
    vs `False`, (3) a solid cube with `surface_depth_um < min_spacing` triggering
    the 1-voxel outer shell fallback, and (4) an all-False volume.
    Expected outcome: envelope covers foreground, surface is a proper shell
    subset, `fill_holes=False` keeps interior cavity empty, thin depth returns
    the exact 1-voxel eroded shell, and empty input yields empty masks.
    """
    vol = np.zeros((30, 30, 30), dtype=bool)
    z, y, x = np.ogrid[:30, :30, :30]
    vol[(z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 10**2] = True

    surface_mask, envelope = generate_morphological_surface_mask(
        vol, pixel_sizes=(1.0, 1.0, 1.0), closing_radius_um=2.0, surface_depth_um=3.0, downscale_factor=1, fill_holes=True
    )
    assert surface_mask.shape == vol.shape and envelope.shape == vol.shape
    assert surface_mask.dtype == bool and envelope.dtype == bool
    assert np.all(envelope[vol])
    assert np.all(envelope[surface_mask])
    assert not surface_mask[15, 15, 15]

    # Hollow cube: fill_holes=True fills the cavity, fill_holes=False preserves it
    hollow = np.zeros((24, 24, 24), dtype=bool)
    hollow[4:20, 4:20, 4:20] = True
    hollow[8:16, 8:16, 8:16] = False
    _, env_filled = generate_morphological_surface_mask(
        hollow, pixel_sizes=ISO_3D, closing_radius_um=1.0, surface_depth_um=2.0, downscale_factor=1, fill_holes=True
    )
    _, env_hollow = generate_morphological_surface_mask(
        hollow, pixel_sizes=ISO_3D, closing_radius_um=1.0, surface_depth_um=2.0, downscale_factor=1, fill_holes=False
    )
    assert env_filled[12, 12, 12]
    assert not env_hollow[12, 12, 12]
    assert env_hollow[4, 12, 12]

    # Thin surface_depth_um (< min voxel spacing) falls back to the immediate 1-voxel outer shell
    cube = np.zeros((16, 16, 16), dtype=bool)
    cube[3:13, 3:13, 3:13] = True
    surf_thin, env_thin = generate_morphological_surface_mask(
        cube,
        pixel_sizes={"Z": 2.0, "Y": 2.0, "X": 2.0},
        closing_radius_um=1.0,
        surface_depth_um=0.5,
        downscale_factor=1,
    )
    expected_shell = env_thin & ~ndi.binary_erosion(env_thin)
    np.testing.assert_array_equal(surf_thin, expected_shell)
    assert surf_thin[3, 8, 8] and not np.any(surf_thin[4:12, 4:12, 4:12])

    # Empty mask handling
    empty_vol = np.zeros((10, 10, 10), dtype=bool)
    surf_empty, env_empty = generate_morphological_surface_mask(empty_vol, pixel_sizes=ISO_3D)
    assert not np.any(surf_empty) and not np.any(env_empty)


def test_morphological_envelope_contains_input_when_downscaled():
    """Verify block max-pool downscaling keeps 100% of foreground voxels inside the envelope.

    Strategy: scatter 150 small ellipsoids in a `(40, 120, 120)` anisotropic mask
    and run `generate_morphological_surface_mask` with `downscale_factor` in `(1, 2, 3)`.
    Expected outcome: zero foreground voxels fall outside `envelope`, and
    `surface` is non-empty with matching shape.
    """
    rng = np.random.default_rng(2)
    shape = (40, 120, 120)
    mask = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for c in rng.integers((3, 4, 4), (37, 116, 116), size=(150, 3)):
        mask |= ((zz - c[0]) / 2) ** 2 + ((yy - c[1]) / 3) ** 2 + ((xx - c[2]) / 3) ** 2 <= 1
    ps = {"Z": 1.0, "Y": 0.5, "X": 0.5}
    for f in (1, 2, 3):
        surface, envelope = generate_morphological_surface_mask(
            mask, pixel_sizes=ps, closing_radius_um=6.0, downscale_factor=f
        )
        assert np.count_nonzero(mask & ~envelope) == 0, f
        assert surface.shape == mask.shape and np.any(surface)


def test_morphological_surface_mask_and_ellipsoid_struct_work_in_2d():
    """Verify 2D masks with 2D pixel sizes work, and other ndims fail with a clear error.

    Strategy: run `generate_morphological_surface_mask` on a `(40, 40)` mask of
    two 20x9 / 20x10 px rectangles split by a 1 px gap, at 0.5 um pixels with a
    1 um closing radius (2 px) and 1 um surface depth, with `downscale_factor`
    1 and 2. It used to build a 3D structuring element and fail with
    'pixel_sizes has 2 entries, expected 3'. `create_ellipsoid_struct(ndim=2)`
    builds the 2D disk.
    Expected outcome: the closing bridges the gap, the envelope contains the
    mask, the surface is the 2 px (<= 1 um) rim of the envelope and excludes
    its interior; the 2D struct of radius 1 um at 0.5 um is a 5x5 disk.
    Negative controls: `create_ellipsoid_struct` still defaults to 3D (a 2-entry
    pixel size raises), and a 4D mask raises a ValueError naming 2D and 3D.
    """
    ps = {"Y": 0.5, "X": 0.5}
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:19] = True
    mask[10:30, 20:30] = True

    surface, envelope = generate_morphological_surface_mask(
        mask, pixel_sizes=ps, closing_radius_um=1.0, surface_depth_um=1.0, downscale_factor=1
    )
    assert surface.shape == mask.shape and envelope.shape == mask.shape
    assert envelope[20, 19] and np.all(envelope[mask])
    assert surface[10, 20] and surface[11, 20] and not surface[12, 20]
    assert not surface[20, 19] and not np.any(surface & ~envelope)

    surface_f2, envelope_f2 = generate_morphological_surface_mask(
        mask, pixel_sizes=ps, closing_radius_um=1.0, surface_depth_um=1.0, downscale_factor=2
    )
    assert np.all(envelope_f2[mask]) and np.any(surface_f2) and not surface_f2[20, 20]

    disk = create_ellipsoid_struct(1.0, pixel_sizes=ps, ndim=2)
    assert disk.shape == (5, 5) and disk[2, 0] and disk[1, 1] and not disk[0, 0]
    tiny = create_ellipsoid_struct(1e-8, pixel_sizes=ps, ndim=2)
    assert tiny.shape == (3, 3) and tiny[1, 1] and tiny.sum() == 1

    with pytest.raises(ValueError, match="expected 3"):
        create_ellipsoid_struct(1.0, pixel_sizes=(0.5, 0.5))
    with pytest.raises(ValueError, match="2D .* or 3D"):
        generate_morphological_surface_mask(np.ones((2, 3, 4, 5), bool), pixel_sizes=(1.0, 1.0, 1.0, 1.0))


def test_estimate_inter_label_distance():
    """Verify isotropic, anisotropic, few-label, and all-zero inter-label distance estimation.

    Strategy: evaluate `estimate_inter_label_distance` on (1) 3 collinear labels
    spaced 10 px apart isotropically, (2) 2 Z-stacked labels 4 slices apart at
    `Z=3 um` (`12 um`), (3) a single label (`< k_nearest`), and (4) an all-zero volume.
    Expected outcome: physical medians and `recommended_closing_um` match exact
    centroid distances, and under-populated/empty volumes fall back cleanly to `min(spacing)`.
    """
    labels = np.zeros((50, 50, 50), dtype=int)
    labels[8:12, 8:12, 8:12] = 1
    labels[8:12, 8:12, 18:22] = 2
    labels[8:12, 8:12, 28:32] = 3

    metrics = estimate_inter_label_distance(labels, pixel_sizes=ISO_3D, k_nearest=2)
    assert {"median_um", "p75_um", "p90_um", "recommended_closing_um"} <= metrics.keys()
    assert np.isclose(metrics["median_um"], 10.0, atol=0.5)

    # Anisotropic Z separation: 4 slices * 3.0 um = 12.0 um
    aniso_labels = np.zeros((20, 20, 20), dtype=int)
    aniso_labels[4:6, 8:10, 8:10] = 1
    aniso_labels[8:10, 8:10, 8:10] = 2
    aniso_metrics = estimate_inter_label_distance(
        aniso_labels, pixel_sizes={"Z": 3.0, "Y": 0.5, "X": 0.5}, k_nearest=2
    )
    assert aniso_metrics["median_um"] == pytest.approx(12.0)
    assert aniso_metrics["p75_um"] == pytest.approx(12.0)
    assert aniso_metrics["p90_um"] == pytest.approx(12.0)
    assert aniso_metrics["recommended_closing_um"] == pytest.approx(7.2)

    # Fewer labels than k_nearest falls back to min spacing
    few_labels = np.zeros((20, 20, 20), dtype=int)
    few_labels[5:8, 5:8, 5:8] = 1
    fallback = estimate_inter_label_distance(few_labels, pixel_sizes=(2.0, 2.0, 2.0), k_nearest=2)
    assert fallback["median_um"] == 2.0 and fallback["recommended_closing_um"] == 3.0

    # Empty all-zero labels fall back to min spacing without error
    empty = np.zeros((10, 10, 10), dtype=int)
    empty_metrics = estimate_inter_label_distance(
        empty, pixel_sizes={"Z": 2.0, "Y": 0.4, "X": 0.4}, k_nearest=2
    )
    assert empty_metrics == {
        "median_um": 0.4,
        "p75_um": 0.4,
        "p90_um": 0.4,
        "recommended_closing_um": 0.6,
    }


def test_estimate_inter_label_distance_neighbours_from_all_labels():
    """Verify subsampled query centroids search nearest neighbours across ALL labels, not only the subsample.

    Strategy: build a 30^3 lattice of 2x2x2-voxel nuclei every 4 voxels (`8 um`
    at `2 um` spacing) and query with `sample_size=500`.
    Expected outcome: `median_um` and `p90_um` equal `8.0`, `recommended_closing_um`
    equals `4.8`, and identical `seed` produces identical results.
    """
    n, step = 30, 4
    labels = np.zeros((n * step,) * 3, dtype=np.int32)
    ids = np.arange(1, n**3 + 1, dtype=np.int32).reshape((n,) * 3)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                labels[dz::step, dy::step, dx::step] = ids
    ps = {"Z": 2.0, "Y": 2.0, "X": 2.0}
    m = estimate_inter_label_distance(labels, pixel_sizes=ps, sample_size=500)
    assert m["median_um"] == pytest.approx(8.0)
    assert m["p90_um"] == pytest.approx(8.0)
    assert m["recommended_closing_um"] == pytest.approx(4.8)
    assert estimate_inter_label_distance(labels, pixel_sizes=ps, seed=3) == estimate_inter_label_distance(
        labels, pixel_sizes=ps, seed=3
    )


# --- R7-10 / R7-11 / R7-14 / R7-15 / R7-09 / R8-18 (processing-geometry) ---


def _ramp_and_grid():
    """A Z ramp (value 10 * z) and a flat 4x4 surface grid at z = 10."""
    ramp = np.broadcast_to(np.arange(20, dtype=float)[:, None, None] * 10.0, (20, 10, 10)).copy()
    yy, xx = np.meshgrid(np.arange(3, 7), np.arange(3, 7), indexing="ij")
    grid = np.stack([np.full_like(yy, 10.0), yy, xx], axis=-1).astype(float)
    return ramp, grid


def test_surface_normal_sampling_interpolation_vocabulary():
    """R7-10: interpolation takes 'nearest' | 'linear' | 'cubic'; 'bicubic' is deprecated; others raise.

    Strategy: sample a Z ramp 0.3 voxel either side of a flat surface.
    Expected outcome: 'linear' and 'cubic' give [97, 100, 103] (they silently fell
    back to 'nearest' = 100 before); 'bicubic' warns (FutureWarning) and equals
    'cubic'; an unknown name raises ValueError.
    """
    ramp, grid = _ramp_and_grid()
    kw = dict(thickness=0.6, num_steps=3, pixel_sizes=ISO_3D)
    near = sample_intensity_along_surface_normals(ramp, grid, interpolation="nearest", **kw)
    np.testing.assert_allclose(near, 100.0)
    for method in ("linear", "cubic"):
        out = sample_intensity_along_surface_normals(ramp, grid, interpolation=method, **kw)
        np.testing.assert_allclose(out[1, 1, :], [97.0, 100.0, 103.0], atol=1e-2)
    with pytest.warns(FutureWarning, match=r"bicubic.*0\.1\.0.*cubic"):
        legacy = sample_intensity_along_surface_normals(ramp, grid, interpolation="bicubic", **kw)
    np.testing.assert_array_equal(
        legacy, sample_intensity_along_surface_normals(ramp, grid, interpolation="cubic", **kw)
    )
    with pytest.raises(ValueError, match="interpolation"):
        sample_intensity_along_surface_normals(ramp, grid, interpolation="spline", **kw)


def test_surface_normal_sampling_cval_and_thin_grid():
    """R7-10: out-of-volume samples take `cval` (default 0); a 1-row grid raises a clear ValueError.

    Expected outcome: with cval=np.nan the samples above the top face are NaN (they
    were indistinguishable from real zeros); a (1, N, 3) grid raises ValueError
    naming the grid shape instead of crashing inside np.gradient.
    """
    ramp, grid = _ramp_and_grid()
    top = grid.copy()
    top[..., 0] = 0.0
    kw = dict(thickness=4.0, num_steps=5, pixel_sizes=ISO_3D, interpolation="linear")
    out0 = sample_intensity_along_surface_normals(ramp, top, **kw)
    out_nan = sample_intensity_along_surface_normals(ramp, top, cval=np.nan, **kw)
    outside = np.isnan(out_nan)
    assert outside.any() and not outside.all()
    np.testing.assert_array_equal(out0[outside], 0.0)
    np.testing.assert_array_equal(out0[~outside], out_nan[~outside])
    with pytest.raises(ValueError, match="at least 2"):
        sample_intensity_along_surface_normals(ramp, grid[:1], **kw)


def test_sample_volume_plane_method_vocabulary():
    """R7-10: method takes 'nearest' | 'linear' | 'cubic'; 'nearest' is order 0; others raise.

    Strategy: sample a Z ramp on a plane at z = 4.3 ('nearest' used to fall back to linear).
    Expected outcome: nearest = 40, linear = 43; an unknown name raises ValueError;
    cval sets the out-of-volume value.
    """
    vol = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None, None] * 10.0, (10, 8, 8)).copy()
    kw = dict(pixel_sizes=ISO_3D, p0_phys=np.array([4.3, 3.5, 3.5]), normal_phys=np.array([1.0, 0, 0]),
              u_range_um=(-2, 2), v_range_um=(-2, 2), u_res=5, v_res=5)
    near, _ = sample_volume_plane(vol, method="nearest", **kw)
    lin, _ = sample_volume_plane(vol, method="linear", **kw)
    np.testing.assert_allclose(near, 40.0)
    np.testing.assert_allclose(lin, 43.0, atol=1e-4)
    with pytest.raises(ValueError, match="method"):
        sample_volume_plane(vol, method="bilinear", **kw)
    far = dict(kw, p0_phys=np.array([40.0, 3.5, 3.5]))
    for method in ("nearest", "linear", "cubic"):
        out, _ = sample_volume_plane(vol, method=method, cval=-1.0, **far)
        np.testing.assert_array_equal(out, -1.0)


def test_fit_plane_ransac_seed_is_reproducible():
    """R7-11: fit_plane_ransac(seed=...) makes the result reproducible; seed=None stays random.

    Strategy: a noisy plane with 30% outliers, max_iters=50 (the R7-11 repro gave
    p0_z in 9.86 ... 10.12 over five unseeded calls).
    Expected outcome: equal seeds give bit-identical results; a Generator is accepted.
    """
    rng = np.random.default_rng(3)
    yy, xx = rng.uniform(0, 40, size=(2, 200))
    zz = 10 + 0.1 * yy + rng.normal(0, 0.3, 200)
    zz[:60] = rng.uniform(0, 30, 60)
    pts = np.column_stack([zz, yy, xx])
    a = fit_plane_ransac(pts, pixel_sizes=ISO_3D, max_iters=50, seed=7)
    b = fit_plane_ransac(pts, pixel_sizes=ISO_3D, max_iters=50, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    c = fit_plane_ransac(pts, pixel_sizes=ISO_3D, max_iters=50, seed=np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], c[0])


def test_fit_plane_ransac_degeneracy_test_is_unit_independent():
    """R7-11: the degeneracy test is relative, so tiny (metre-scale) pixel sizes still fit.

    Strategy: the same plane with pixel sizes of 1 µm and of 1e-6 (as if in metres),
    thresholds scaled alike (the absolute area cutoff 1e-6 rejected every triple).
    Expected outcome: the same unit normal, and p0 scaled by 1e-6; exactly collinear
    points still raise RuntimeError.
    """
    rng = np.random.default_rng(0)
    yy, xx = rng.uniform(0, 20, size=(2, 50))
    pts = np.column_stack([5 + 0.2 * yy - 0.1 * xx, yy, xx])
    p0, n = fit_plane_ransac(pts, pixel_sizes=(1.0, 1.0, 1.0), inlier_threshold_um=0.1, seed=0)
    p0_m, n_m = fit_plane_ransac(pts, pixel_sizes=(1e-6,) * 3, inlier_threshold_um=1e-7, seed=0)
    np.testing.assert_allclose(n_m, n, atol=1e-9)
    np.testing.assert_allclose(p0_m, p0 * 1e-6, rtol=1e-9)
    collinear = np.column_stack([np.arange(10.0), np.arange(10.0), np.arange(10.0)])
    with pytest.raises(RuntimeError):
        fit_plane_ransac(collinear, pixel_sizes=ISO_3D, max_iters=50, seed=0)


def test_windowed_slice_projection_even_window_does_not_print(capsys):
    """R7-09: an even window is widened to the next odd size silently (documented), no print."""
    vol = np.random.default_rng(0).random((8, 5, 5))
    out = windowed_slice_projection(vol, window_size=4, pixel_sizes=ISO_3D)
    np.testing.assert_array_equal(out, windowed_slice_projection(vol, window_size=5, pixel_sizes=ISO_3D))
    assert capsys.readouterr().out == ""


def test_optimized_entire_labels_touching_mask_without_isin_matches_reference():
    """R7-15: dropping the always-true np.isin(dilated, labels) keeps the result.

    Strategy: compare against a direct reference (labels whose expansion touches the mask).
    """
    from skimage.segmentation import expand_labels

    rng = np.random.default_rng(1)
    labels = np.zeros((12, 30, 30), dtype=np.int32)
    for i in range(1, 9):
        z, y, x = rng.integers(2, [10, 28, 28])
        labels[z - 1:z + 2, y - 2:y + 3, x - 2:x + 3] = i
    mask = np.zeros(labels.shape, bool)
    mask[:, :, :6] = True
    out = optimized_entire_labels_touching_mask(labels, mask, distance=2, pixel_sizes=ISO_3D)
    touching = np.unique(expand_labels(labels, 2)[mask])
    touching = touching[touching > 0]
    np.testing.assert_array_equal(out, labels * np.isin(labels, touching))


@pytest.mark.parametrize(
    "call",
    [
        lambda: create_ellipsoid_struct(2.0, spacing=(1.0, 1.0, 1.0)),
        lambda: create_ellipsoid_struct(2.0, (1.0, 1.0, 1.0)),
        lambda: generate_morphological_surface_mask(np.ones((6, 6, 6), bool), spacing=(1.0, 1.0, 1.0)),
        lambda: estimate_inter_label_distance(np.ones((4, 4, 4), int), spacing=(1.0, 1.0, 1.0)),
    ],
    ids=["ellipsoid_kw", "ellipsoid_positional", "surface_mask", "inter_label"],
)
def test_spacing_alias_is_deprecated(call):
    """R8-18 / D08: `spacing=` (keyword or positional) is a deprecated alias of `pixel_sizes=`.

    Expected outcome: a FutureWarning naming spacing, 0.1.0 and pixel_sizes,
    attributed to this test file (the caller), not to the library.
    """
    with pytest.warns(FutureWarning, match=r"spacing.*0\.1\.0.*pixel_sizes") as rec:
        call()
    assert any(r.filename == __file__ for r in rec if issubclass(r.category, FutureWarning))
