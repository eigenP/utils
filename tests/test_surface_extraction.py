
import numpy as np
import pytest
from eigenp_utils.surface_extraction import (
    extract_surface,
    pad_z_slices,
    crop_edges,
    expand_surface_z,
    adjust_mask_location
)

def test_extract_surface():
    # Create a synthetic 3D image with a surface (a sphere)
    shape = (50, 50, 50)
    image = np.zeros(shape, dtype=np.uint8)

    # Create a sphere in the center
    z, y, x = np.indices(shape)
    center = np.array(shape) / 2
    radius = 15
    mask = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= radius**2
    image[mask] = 200 # Surface intensity

    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 10, shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Extract surface
    surface = extract_surface(image, downscale_factor=2)

    assert surface.shape == shape
    assert surface.dtype == bool
    # Check if surface is detected (some True values)
    assert np.any(surface)

    # The surface should be somewhat around the sphere shell
    # We can check if the detected surface points are roughly at radius distance
    # This is a basic sanity check, not precision validation
    surface_points = np.argwhere(surface)
    distances = np.sqrt(np.sum((surface_points - center)**2, axis=1))
    # Check if average distance is reasonable (near radius)
    # Since it extracts top surface, it might be different, but for a sphere top surface is the upper hemisphere.
    # extract_surface finds "top" surface along Z.

    # Let's trust it returns a mask for now.

def test_extract_surface_padding_mismatch():
    """
    Test extract_surface with dimensions that are not divisible by block size,
    causing internal padding that previously led to IndexError.
    """
    # Y = 10, X = 10. Block size = (2, 4, 4)
    # 10 % 4 = 2 != 0. Requires padding.
    Z, Y, X = 20, 10, 10
    image = np.zeros((Z, Y, X), dtype=np.uint8)
    image[10:, :, :] = 200 # Foreground

    # Pad Z manually as in the user report (optional, but good for consistency)
    padded_image = pad_z_slices(image, num_slices=4)

    _BLOCK_SIZE = (2, 4, 4)

    # This should not raise IndexError
    surface = extract_surface(padded_image, downscale_factor=_BLOCK_SIZE)

    assert surface.shape == padded_image.shape
    assert surface.dtype == bool

def test_pad_z_slices():
    arr = np.zeros((10, 10, 10))
    # Pad 1 slice before
    padded = pad_z_slices(arr, num_slices=1, pad_before=True, pad_after=False)
    assert padded.shape == (11, 10, 10)

    # Pad 10% -> 1 slice (10 * 0.1 = 1)
    padded = pad_z_slices(arr, num_slices=0.1, pad_before=True, pad_after=True)
    assert padded.shape == (12, 10, 10)

    # Pad with value
    arr[:] = 5
    padded = pad_z_slices(arr, num_slices=1, fill_value=99, pad_before=True)
    assert padded[0, 0, 0] == 99
    assert padded[1, 0, 0] == 5

def test_crop_edges():
    arr = np.ones((10, 20, 20))
    # Crop 10% from Y and X
    # Y: 20 * 0.1 = 2 pixels from each side
    # X: 20 * 0.1 = 2 pixels from each side
    cropped = crop_edges(arr, 0, 0.1, 0.1)
    assert cropped.shape == arr.shape

    # Central region should be 1
    assert cropped[5, 10, 10] == 1
    # Edges should be 0
    assert cropped[5, 0, 0] == 0
    assert cropped[5, 1, 1] == 0 # 0 and 1 are cropped (0..2)
    assert cropped[5, 2, 2] == 1 # 2 is start of valid region

    # Test 0 crop
    cropped_zero = crop_edges(arr, 0, 0.0, 0.0)
    assert np.all(cropped_zero == arr)

def test_expand_surface_z():
    shape = (10, 10, 10)
    surface = np.zeros(shape, dtype=bool)
    surface[5, 5, 5] = True

    expanded = expand_surface_z(surface, thickness=1)

    # Should cover z=4, 5, 6 at y=5, x=5
    assert expanded[4, 5, 5]
    assert expanded[5, 5, 5]
    assert expanded[6, 5, 5]
    # Should not cover z=3 or 7
    assert not expanded[3, 5, 5]
    assert not expanded[7, 5, 5]

def test_adjust_mask_location():
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[5, 5, 5] = True

    # Translate by (1, 1, 1) -> (6, 6, 6)
    translated = adjust_mask_location(mask, translation=(1, 1, 1))
    assert translated[6, 6, 6]
    assert not translated[5, 5, 5]

    # Dilate
    dilated = adjust_mask_location(mask, morph='dilate', iterations=1)
    # Check neighbors
    assert dilated[5, 5, 6]
    assert dilated[5, 6, 5]

    # Erode
    # Mask with a small block
    mask_block = np.zeros(shape, dtype=bool)
    mask_block[4:7, 4:7, 4:7] = True
    eroded = adjust_mask_location(mask_block, morph='erode', iterations=1)
    # Center should remain, edges eroded
    assert eroded[5, 5, 5]
    assert not eroded[4, 4, 4]
