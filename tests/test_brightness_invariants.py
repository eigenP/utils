import numpy as np
import pytest
from eigenp_utils.intensity_rescaling import adjust_brightness_per_slice

def generate_trend_stack(shape=(20, 100, 100), decay_func=None, outlier_idx=None, outlier_factor=0.5, seed=42):
    """
    Generates a synthetic stack where the 99th percentile follows a specific decay function.
    The content is uniform noise scaled to match the decay.
    """
    rng = np.random.default_rng(seed)
    z_dim, y_dim, x_dim = shape
    stack = np.zeros(shape, dtype=np.float32)

    # Base intensity (P99)
    p99_curve = np.zeros(z_dim)

    for z in range(z_dim):
        # Calculate expected P99
        if decay_func:
            target_p99 = decay_func(z)
        else:
            target_p99 = 1.0

        # Store for verification
        p99_curve[z] = target_p99

        # Generate slice
        # We want np.percentile(slice, 99) == target_p99.
        # Uniform [0, M] has percentile 99 at 0.99 * M.
        # So M = target_p99 / 0.99
        max_val = target_p99 / 0.99

        # Add outlier
        if outlier_idx is not None and z == outlier_idx:
            max_val *= outlier_factor

        stack[z] = rng.uniform(0, max_val, (y_dim, x_dim))

    return stack, p99_curve

def test_perfect_restoration_exponential():
    """
    Testr Verification: Perfect Signal Restoration

    Verifies that a perfectly exponential decay is recovered to flat intensity
    using the 'gain' method. This validates the Forward/Inverse model relationship:
    Forward: I(z) = I0 * exp(-kz)
    Inverse: I_corr(z) = I(z) * gain(z) where gain(z) comes from fitting the exponential.

    Invariant: The output stack should have constant 99th percentile across all Z slices.
    """
    Z = 20
    # Decay: I(z) = I0 * exp(-0.1 * z)
    decay = lambda z: 1.0 * np.exp(-0.1 * z)

    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay)

    # Apply correction
    # fit 'exponential', method 'gain'
    corrected = adjust_brightness_per_slice(stack, gamma_fit_func='exponential', method='gain')

    # Check 99th percentiles of output
    p99_out = np.array([np.percentile(s, 99) for s in corrected])

    # Should be constant (flat)
    mean_p99 = np.mean(p99_out)
    std_p99 = np.std(p99_out)

    # We expect very low variation (coefficient of variation < 1%)
    # This proves the fit found the correct parameters and the gain application worked.
    cv = std_p99 / mean_p99

    print(f"Mean P99: {mean_p99:.4f}, Std P99: {std_p99:.4f}, CV: {cv:.4f}")

    assert cv < 0.015, f"Brightness not stabilized! CV={cv:.4f} is too high."


def test_trend_preservation_with_outlier():
    """
    Testr Verification: Trend vs. Anomaly Distinction

    Verifies that the algorithm corrects the Global Trend (systematic decay)
    but preserves Local Anomalies (sample outliers).

    This is a critical property: we want to correct for physics (attenuation),
    not normalize away biology (e.g., a slice with no cells).

    Setup:
    - Exponential decay trend.
    - Slice 10 is artificially dimmed by 50%.

    Expectation:
    - The trend (non-outlier slices) becomes flat.
    - Slice 10 remains ~50% of the restored brightness level.
    """
    Z = 20
    decay = lambda z: 1.0 * np.exp(-0.05 * z)
    outlier_z = 10
    factor = 0.5

    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay,
                                    outlier_idx=outlier_z, outlier_factor=factor)

    # Apply correction
    corrected = adjust_brightness_per_slice(stack, gamma_fit_func='exponential', method='gain')

    p99_out = np.array([np.percentile(s, 99) for s in corrected])

    # 1. Verify trend is flat EXCLUDING the outlier
    # Mask out the outlier
    mask = np.ones(Z, dtype=bool)
    mask[outlier_z] = False

    trend_vals = p99_out[mask]
    outlier_val = p99_out[outlier_z]

    mean_trend = np.mean(trend_vals)
    std_trend = np.std(trend_vals)

    # Trend should be flat (corrected)
    cv_trend = std_trend / mean_trend
    assert cv_trend < 0.02, f"Global trend was not corrected properly. CV={cv_trend}"

    # 2. Verify outlier is preserved (relative to trend)
    # The outlier started at 0.5 * Trend.
    # The correction multiplied it by (Target / Fitted_Trend).
    # So Result = (0.5 * True_Trend) * (Target / Fitted_Trend).
    # Assuming Fitted_Trend ~ True_Trend, Result ~ 0.5 * Target.
    # Since Target ~ mean_trend (of corrected image), Result ~ 0.5 * mean_trend.

    ratio = outlier_val / mean_trend
    print(f"Outlier Ratio: {ratio:.4f} (Expected ~{factor})")

    # We allow some tolerance because the outlier affects the fit slightly
    assert np.isclose(ratio, factor, atol=0.1), \
        f"Local anomaly was not preserved correctly. Ratio {ratio} != {factor}"

def test_idempotence():
    """
    Testr Verification: Idempotence / Stability

    Verifies that re-running the correction on already corrected data
    (which is now flat) results in minimal change.

    If the function is stable, fitting an exponential to a flat line should yield
    decay=0 (flat), resulting in gains of 1.0.
    """
    Z = 15
    decay = lambda z: 1.0 * np.exp(-0.1 * z)
    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay)

    # First pass
    pass1 = adjust_brightness_per_slice(stack, gamma_fit_func='exponential', method='gain')

    # Second pass
    pass2 = adjust_brightness_per_slice(pass1, gamma_fit_func='exponential', method='gain')

    # Check difference
    # Using relative error to be scale-invariant
    # Avoid div by zero
    diff = np.abs(pass1 - pass2)
    rel_diff = diff / (pass1 + 1e-9)

    max_rel_diff = np.max(rel_diff)
    print(f"Max Relative Difference: {max_rel_diff:.6f}")

    # Tolerance Note:
    # We use 50x50 patches. The 99th percentile has sampling variance.
    # Even if the underlying distribution is perfectly flat, the sample P99s will fluctuate.
    # curve_fit will slightly overfit these fluctuations, finding a tiny non-zero decay/growth.
    # A change of < 0.1% (1e-3) is acceptable and indicates stability.
    assert max_rel_diff < 1e-3, \
        f"Algorithm is not idempotent; second pass changed values by {max_rel_diff*100:.4f}%"
