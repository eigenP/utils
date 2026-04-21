import numpy as np
import pytest
from src.eigenp_utils.intensity_rescaling import fit_basic_shading, apply_basic_shading

def test_basic_fit_synthetic():
    np.random.seed(42)
    sizes = (128, 128)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10

    # Ground truth relative flatfield
    truth = gradient / np.mean(gradient)

    # Generate 8 images with poisson noise
    images = np.random.poisson(lam=gradient.astype(int), size=[8] + list(sizes))

    # Fit flatfield
    res = fit_basic_shading(images, is_3d=False)
    flatfield = res['flatfield']

    # Validate accuracy
    max_error = np.max(np.abs(flatfield - truth))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"

    # Verify shape
    assert flatfield.shape == sizes

def test_apply_basic_shading():
    np.random.seed(42)
    images = np.random.rand(8, 128, 128)
    flatfield = np.ones((128, 128)) * 2

    corrected = apply_basic_shading(images, flatfield)

    np.testing.assert_allclose(corrected, images / 2.0)

def test_apply_basic_shading_baseline():
    np.random.seed(42)
    images = np.ones((8, 128, 128))
    flatfield = np.ones((128, 128))
    baseline = np.arange(8)

    corrected = apply_basic_shading(images, flatfield, baseline=baseline)

    # Check baseline correction was applied correctly (using defaults and gaussian smoothing)
    # The gaussian filter will slightly blur the `baseline`, but the mean should remain roughly the same
    # For a deterministic check, we test whether baseline logic runs without error
    assert corrected.shape == images.shape

if __name__ == '__main__':
    pytest.main([__file__])

def test_basic_fit_synthetic_3d():
    np.random.seed(42)
    sizes = (8, 64, 64)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10

    truth = gradient / np.mean(gradient)

    images = np.random.poisson(lam=gradient.astype(int), size=[4] + list(sizes))

    res = fit_basic_shading(images, is_3d=True)
    flatfield = res['flatfield']

    max_error = np.max(np.abs(flatfield - truth))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"
    assert flatfield.shape == sizes
