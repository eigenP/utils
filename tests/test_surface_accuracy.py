
import numpy as np
import pytest
from eigenp_utils.surface_extraction import extract_surface

def generate_synthetic_landscape(shape, z_offset=20, amplitude=5, period=50):
    """
    Generates a binary volume with a sinusoidal 'ground'.
    Returns the volume and the ground truth height map.
    """
    d, h, w = shape
    z, y, x = np.indices(shape)

    # Surface function: Z = z_offset + A * sin(x) + A * cos(y)
    # This creates an egg-carton-like surface
    surface_height = z_offset + amplitude * np.sin(2 * np.pi * x / period) + amplitude * np.cos(2 * np.pi * y / period)

    # Create binary volume
    # Pixels with z >= surface_height are "Ground" (Foreground)
    # Pixels with z < surface_height are "Air" (Background)
    # extract_surface finds the "top" of the foreground, which corresponds to the smallest z index where mask is True.

    volume = np.zeros(shape, dtype=np.uint8)
    mask = z >= surface_height
    volume[mask] = 200 # Foreground intensity

    return volume, surface_height

def test_surface_accuracy():
    """
    Verifies that extracted surface follows the ground truth geometry.
    This checks functional correctness: does the algorithm actually find the surface?
    """
    shape = (64, 128, 128)
    # Period must be large enough relative to downscale * sigma to avoid smoothing it out
    # Downscale=2, Sigma=1.0 -> Effective smoothing sigma ~ 2-3 pixels. Period 64 is plenty.
    image, gt_height = generate_synthetic_landscape(shape, z_offset=32, amplitude=5, period=64)

    # Run extraction
    # Using low sigma to preserve the shape for accuracy testing
    surface_mask = extract_surface(image, downscale_factor=2, gaussian_sigma=1.0)

    # Reconstruct height map from mask (Find first True along Z)
    # If a column has no True, argmax is 0. But our surface is around z=32, so 0 is far away.
    # We should ensure we don't include those if any.
    has_surface = np.any(surface_mask, axis=0)
    assert np.all(has_surface), "Surface should be detected everywhere"

    extracted_z = np.argmax(surface_mask, axis=0)

    # Calculate Mean Absolute Error
    diff = extracted_z - gt_height
    mae = np.mean(np.abs(diff))

    print(f"Accuracy MAE: {mae:.4f}")

    # Allow small error due to binning, smoothing, and integer quantization
    assert mae < 1.5, f"Surface extraction MAE {mae} is too high (expected < 1.5)"

def test_translation_invariance():
    """
    Verifies that shifting the input Z shifts the output Z.
    This is a critical invariant for 3D processing pipelines.
    """
    shape = (100, 64, 64) # Taller Z to allow shift without clipping interesting parts
    # Create a distinct surface
    image, _ = generate_synthetic_landscape(shape, z_offset=40, amplitude=5, period=40)

    shift_amount = 8 # Integer shift

    # Case 1: Original
    mask1 = extract_surface(image, downscale_factor=2, gaussian_sigma=1.0)
    z1 = np.argmax(mask1, axis=0)

    # Case 2: Shifted Image
    # Shift data 'down' (to higher indices), filling top with 0
    image_shifted = np.roll(image, shift_amount, axis=0)
    image_shifted[:shift_amount] = 0

    mask2 = extract_surface(image_shifted, downscale_factor=2, gaussian_sigma=1.0)
    z2 = np.argmax(mask2, axis=0)

    # Compare
    # calculate shift for each pixel
    detected_shift = z2 - z1

    # Statistics
    mean_shift = np.mean(detected_shift)
    std_shift = np.std(detected_shift)

    print(f"Mean Shift: {mean_shift:.4f}, Std Shift: {std_shift:.4f}")

    # The mean shift should match the applied shift
    assert np.abs(mean_shift - shift_amount) < 0.5, f"Expected shift {shift_amount}, got {mean_shift}"

    # The variance of the shift should be low (the shape shouldn't distort)
    assert std_shift < 0.5, f"Shift introduced distortion (std: {std_shift})"
