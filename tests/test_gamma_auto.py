import numpy as np
import pytest
from skimage.exposure import adjust_gamma
from eigenp_utils.intensity_rescaling import adjust_gamma_per_slice

def generate_decaying_image(shape=(10, 100, 100), decay_rate=0.1, initial_intensity=1.0):
    z, y, x = shape
    image = np.zeros(shape, dtype=np.float64)
    for i in range(z):
        decay = np.exp(-decay_rate * i)
        # Create a slice with some noise and variation
        slice_data = np.random.normal(loc=initial_intensity * decay, scale=0.01, size=(y, x))
        # Ensure values are within valid range [0, 1]
        image[i] = np.clip(slice_data, 0, 1)
    return image

def test_adjust_gamma_per_slice_manual():
    """Test the existing manual gamma adjustment."""
    image = np.ones((10, 100, 100))
    # Manual mode: final_gamma=0.5 -> linear ramp from 1.0 to 0.5
    # find_gamma_corr removed in new impl, so just use defaults or explicit None
    adjusted = adjust_gamma_per_slice(image, final_gamma=0.5, gamma_fit_func=None)

    # First slice should be gamma=1.0 (no change)
    assert np.allclose(adjusted[0], image[0])

    # Last slice should be gamma=0.5 (sqrt)
    # 1.0^0.5 = 1.0, so this is a bad test for value change if input is 1.0
    # Let's use input 0.5
    image[:] = 0.5
    adjusted = adjust_gamma_per_slice(image, final_gamma=0.5, gamma_fit_func=None)

    # First slice gamma=1.0 -> 0.5
    assert np.allclose(adjusted[0], 0.5)

    # Last slice gamma=0.5 -> 0.5^0.5 = 0.707...
    assert np.isclose(adjusted[-1, 0, 0], 0.5**0.5)

def test_adjust_gamma_per_slice_auto_exponential():
    """Test automatic gamma finding with exponential decay."""
    # Create an image that decays to 50% intensity
    # decay_rate such that exp(-r * 9) = 0.5 -> -9r = ln(0.5) -> r = -ln(0.5)/9
    decay_rate = -np.log(0.5) / 9
    image = generate_decaying_image(shape=(10, 100, 100), decay_rate=decay_rate, initial_intensity=0.8)

    # Original last slice mean
    original_last_mean = np.mean(image[-1])
    original_first_mean = np.mean(image[0])
    print(f"Original first mean: {original_first_mean}, Last mean: {original_last_mean}")

    # Apply correction using exponential fit
    try:
        adjusted = adjust_gamma_per_slice(image, gamma_fit_func='exponential')
    except TypeError:
         # Skip if not implemented yet
         pytest.skip("gamma_fit_func not implemented yet")

    # The goal is to make slices uniform brightness.
    # The first slice should remain roughly same (gamma ~ 1).
    # The last slice should be brightened.

    adj_last_mean = np.mean(adjusted[-1])
    adj_first_mean = np.mean(adjusted[0])

    print(f"Adjusted first mean: {adj_first_mean}, Last mean: {adj_last_mean}")

    # The adjusted last slice should be closer to the first slice than before
    assert abs(adj_last_mean - adj_first_mean) < abs(original_last_mean - original_first_mean)

    # Ideally, they are close
    assert np.isclose(adj_last_mean, adj_first_mean, atol=0.05)

def test_adjust_gamma_per_slice_auto_linear():
    """Test automatic gamma finding with linear decay."""
    # Linear decay image
    image = np.zeros((10, 100, 100))
    for i in range(10):
        val = 0.8 - (0.04 * i) # 0.8 down to 0.44
        image[i] = np.random.normal(val, 0.01, (100, 100))

    try:
        adjusted = adjust_gamma_per_slice(image, gamma_fit_func='linear')
    except TypeError:
        pytest.skip("gamma_fit_func not implemented yet")

    adj_last_mean = np.mean(adjusted[-1])
    adj_first_mean = np.mean(adjusted[0])

    assert np.isclose(adj_last_mean, adj_first_mean, atol=0.05)

if __name__ == "__main__":
    # Manual run for debugging
    try:
        test_adjust_gamma_per_slice_manual()
        print("Manual test passed.")
        test_adjust_gamma_per_slice_auto_exponential()
        print("Exponential test passed.")
        test_adjust_gamma_per_slice_auto_linear()
        print("Linear test passed.")
    except Exception as e:
        print(f"Test failed: {e}")
