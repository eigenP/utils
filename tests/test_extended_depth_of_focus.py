import numpy as np
import scipy.ndimage
from eigenp_utils.extended_depth_of_focus import best_focus_image
import pytest

def test_focus_sharpness_vs_contrast():
    """
    Verifies that the focus stacking algorithm selects planes based on sharpness (high-frequency content)
    rather than global contrast (variance).

    This guards against a common regression where the metric simplifies to `std(image)`,
    which fails when a blurry region has high contrast (e.g., a steep gradient or strong shadow)
    compared to a sharp region with lower contrast (e.g., fine texture).

    The Invariant:
    Given two images I_sharp and I_blurry, where I_sharp has high Laplacian energy but low variance,
    and I_blurry has low Laplacian energy but high variance, the algorithm MUST choose I_sharp.
    """
    shape = (2, 100, 100)
    stack = np.zeros(shape, dtype=np.float64)

    # Slice 0: "Sharp" but low variance (e.g., checkerboard with small amplitude)
    # Checkerboard frequency = 1/10 pixels
    y, x = np.indices((100, 100))
    checker = ((x // 5) % 2 == (y // 5) % 2).astype(float)
    # Amplitude is 0 to 1. Std ~ 0.5.
    stack[0] = checker

    # Slice 1: "Blurry" but high variance (e.g., strong gradient)
    # Gradient: 0 to ~100. Std ~ 20.
    # Laplacian of a linear gradient is 0 (except at boundaries).
    gradient = (x + y) / 2.0
    stack[1] = gradient

    # Run best_focus_image and get the height map (indices of best focus)
    # We expect the majority of the image to pick index 0.
    _, height_map = best_focus_image(stack, return_heightmap=True)

    median_idx = np.median(height_map)

    # Analytical checks of our inputs to ensure the test case is valid
    std_0 = np.std(stack[0])
    std_1 = np.std(stack[1])
    # Use scipy.ndimage.laplace for ground truth check
    lap_0 = np.std(scipy.ndimage.laplace(stack[0]))
    lap_1 = np.std(scipy.ndimage.laplace(stack[1]))

    # Assert that our test case actually models the "High Contrast Blur vs Low Contrast Sharp" scenario
    assert std_1 > 10 * std_0, "Slice 1 must have significantly higher contrast for the test to be meaningful"
    assert lap_0 > 5 * lap_1, "Slice 0 must have significantly higher sharpness for the test to be meaningful"

    # The actual verification
    assert median_idx == 0, (
        f"Algorithm selected the blurry high-contrast slice (Index {median_idx}) instead of the sharp one. "
        "This indicates reliance on global variance instead of local sharpness."
    )
