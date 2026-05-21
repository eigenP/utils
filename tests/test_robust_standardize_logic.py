import numpy as np
import scipy.stats as stats
import pytest
from eigenp_utils.stats import robust_standardize

def test_robust_standardize_hierarchical_fallback():
    """
    Testr: Verify hierarchical dispersion fallback guarantees in robust standardization.

    Why:
    When standardizing real-world or synthetic data, it's common to encounter zero-inflated
    or highly-tied distributions where >50% of the values are identical. Standard robust
    estimators like Median Absolute Deviation (MAD) collapse to 0 in these cases, causing
    division-by-zero errors or producing infinite scores if blindly regularized with unprincipled
    epsilon values (e.g., mad + 1e-8). This epsilon padding destroys the structural scale of the
    data and artificially explodes the magnitude of outlier deviations.

    How:
    This test verifies that `robust_standardize` correctly implements a mathematically sound
    hierarchical fallback sequence (MAD -> MeanAD -> STD) when a primary estimator collapses,
    preserving the scale logic dynamically rather than falling back to epsilon noise.

    Theory & Catches:
    - **Standard Case:** Normal variation where MAD > 0. Validates that normal scaling is unchanged.
    - **Tied Case (Zero-Inflated):** MAD collapses to 0 because median is highly dominant. MeanAD
      must take over, adjusting BOTH the location (median->mean) and scale to preserve symmetry.
    - **Degenerate/Constant Case:** All values identical. Both MAD and MeanAD collapse. The fallback
      should gracefully yield a constant output array (all zeros) without raising errors.
    - **Dimensionality:** The fallback logic must apply independently per slice/axis.
    """
    # 1. Normal case: MAD works
    x1 = np.array([1, 2, 3, 4, 5, 100], dtype=float)
    # Median = 3.5
    # Absolute deviations from median: 2.5, 1.5, 0.5, 0.5, 1.5, 96.5
    # MAD = np.median([0.5, 0.5, 1.5, 1.5, 2.5, 96.5]) = 1.5
    z1 = robust_standardize(x1)

    mad_scale = 1.5 * (1.0 / stats.norm.ppf(0.75))
    expected_z1 = (x1 - 3.5) / mad_scale
    np.testing.assert_allclose(z1, expected_z1, err_msg="MAD standardization failed for normal variation data")

    # 2. Tied case: MAD fails, MeanAD works
    x2 = np.array([5, 5, 5, 5, 10, 20], dtype=float)
    # Median = 5
    # Absolute deviations from median: 0, 0, 0, 0, 5, 15
    # MAD = np.median([0, 0, 0, 0, 5, 15]) = 0. MAD collapses.
    # Mean = 50 / 6 = 8.333333333333334
    z2 = robust_standardize(x2)

    mean = np.mean(x2)
    mean_ad = np.mean(np.abs(x2 - mean))
    mean_ad_scale = mean_ad * np.sqrt(np.pi / 2.0)
    expected_z2 = (x2 - mean) / mean_ad_scale
    np.testing.assert_allclose(z2, expected_z2, err_msg="MeanAD fallback failed for highly tied data where MAD collapses")

    # 3. Constant case: all estimators fail
    x3 = np.array([5, 5, 5, 5, 5, 5], dtype=float)
    # MAD = 0, MeanAD = 0, STD = 0
    z3 = robust_standardize(x3)
    np.testing.assert_allclose(z3, np.zeros_like(x3), err_msg="Constant fallback failed for identical values")

    # 4. Multidimensional fallback testing (axis=1)
    # Verify that fallbacks trigger per-slice independently
    x_multi = np.vstack([x1, x2, x3])
    z_multi = robust_standardize(x_multi, axis=1)

    np.testing.assert_allclose(z_multi[0], expected_z1, err_msg="Multidimensional MAD slice failed")
    np.testing.assert_allclose(z_multi[1], expected_z2, err_msg="Multidimensional MeanAD fallback slice failed")
    np.testing.assert_allclose(z_multi[2], np.zeros_like(x3), err_msg="Multidimensional Constant fallback slice failed")
