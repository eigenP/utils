
import numpy as np
import pytest
from eigenp_utils.surface_extraction import extract_surface
from scipy.ndimage import gaussian_filter

def generate_sine_wave_volume(shape, amplitude=10, period=50, base_height=25, noise_sigma=0.0):
    """
    Generates a 3D volume with a sine wave surface.
    Z, Y, X order.
    Surface is defined as z = base_height + amplitude * sin(2*pi*x / period).
    """
    Z, Y, X = shape
    volume = np.zeros(shape, dtype=np.uint8)

    # Grid
    x = np.arange(X)

    # Surface function Z(x) - independent of Y for simplicity
    surface_z = base_height + amplitude * np.sin(2 * np.pi * x / period)

    # Create volume
    # Voxels where z >= surface_z are foreground (high intensity)
    # Using a soft edge or hard edge? Let's use hard edge + smoothing.

    for i in range(X):
        z_cutoff = int(np.round(surface_z[i]))
        z_cutoff = max(0, min(Z, z_cutoff))

        # Fill from z_cutoff to Z (assuming we are looking for the 'top' surface,
        # which extract_surface defines as the first index from 0.
        # So we should fill from index z_cutoff to end?
        # Let's verify 'top'. extract_surface does argmax.
        # If we fill 255 at z >= z_cutoff, argmax will return z_cutoff.
        volume[z_cutoff:, :, i] = 255

    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_sigma, shape)
        volume = np.clip(volume.astype(float) + noise, 0, 255).astype(np.uint8)

    return volume, surface_z

def test_sine_wave_reconstruction():
    """
    Testr 🔎 Verification: Surface Reconstruction Accuracy

    Verifies that extract_surface can accurately recover a smooth analytical surface (sine wave).

    The Invariant:
    The recovered height map (argmax of mask) should match the ground truth function
    within a reasonable tolerance (Mean Absolute Error < 2.0 pixels).

    This ensures:
    1. The downsampling/upsampling logic (using order=3 bicubic) preserves shape.
    2. The smoothing (Gaussian) doesn't flatten the signal excessively.
    3. The coordinate system is consistent.
    """

    # 1. Setup
    shape = (60, 100, 100) # Z, Y, X
    amp = 10
    period = 40
    # Create sine wave
    volume, gt_profile = generate_sine_wave_volume(shape, amplitude=amp, period=period, base_height=30)

    # 2. Run Algorithm
    # Use downscale_factor=2 to test the interpolation logic.
    mask = extract_surface(volume, downscale_factor=2, gaussian_sigma=1.0)

    # 3. Recover Height Map from Mask
    # Mask is Z, Y, X.
    # Find first True along Z.

    # We need to handle columns where no surface was found (though our volume is full).
    # But argmax returns 0 if all False.
    detected_z = np.argmax(mask, axis=0)

    # Check coverage (should be 100%)
    has_surface = np.any(mask, axis=0)
    assert np.mean(has_surface) > 0.99, "Surface should be detected almost everywhere."

    # 4. Compare with Ground Truth
    # gt_profile is 1D array of length X.
    # detected_z is (Y, X).
    # We average detected_z over Y since the surface is constant in Y.

    avg_detected_profile = np.mean(detected_z, axis=0)

    # Calculate Error
    # Note: ground truth is float, detected is int (indices).
    # We expect some quantization error + smoothing error.

    diff = avg_detected_profile - gt_profile
    mae = np.mean(np.abs(diff))

    print(f"\nMean Absolute Error: {mae:.4f} pixels")
    print(f"Max Error: {np.max(np.abs(diff)):.4f} pixels")

    # Tolerance:
    # Downscaling by 2 means precision loss. Smoothing shifts edges.
    # MAE < 2.0 is a reasonable goal for a 10px amplitude wave.
    assert mae < 2.0, f"Reconstruction error too high (MAE={mae:.4f})"

    # Correlation check (shape preservation)
    corr = np.corrcoef(avg_detected_profile, gt_profile)[0, 1]
    print(f"Correlation: {corr:.4f}")
    assert corr > 0.98, "Reconstructed shape does not match sine wave."

def test_translation_invariance():
    """
    Testr 🔎 Verification: Surface Translation Invariance

    Verifies that shifting the input volume by K pixels shifts the output surface by K pixels.
    """
    shape = (60, 50, 50)
    # Generate base volume
    vol1, _ = generate_sine_wave_volume(shape, base_height=20)

    # Generate shifted volume (+10 pixels)
    vol2, _ = generate_sine_wave_volume(shape, base_height=30)

    # Run extraction
    # Use smaller sigma to reduce edge shift sensitivity to threshold changes
    mask1 = extract_surface(vol1, downscale_factor=2, gaussian_sigma=1.0)
    mask2 = extract_surface(vol2, downscale_factor=2, gaussian_sigma=1.0)

    z1 = np.argmax(mask1, axis=0).astype(float)
    z2 = np.argmax(mask2, axis=0).astype(float)

    # Filter valid
    valid = (np.any(mask1, axis=0)) & (np.any(mask2, axis=0))

    diff = z2[valid] - z1[valid]
    mean_shift = np.mean(diff)
    std_shift = np.std(diff)

    print(f"\nMean Shift: {mean_shift:.4f} (Expected 10.0)")
    print(f"Std Shift: {std_shift:.4f}")

    # Check
    assert np.abs(mean_shift - 10.0) < 1.0, f"Shift not preserved. Mean shift: {mean_shift}"
    assert std_shift < 1.0, "Shift should be uniform across the surface."
