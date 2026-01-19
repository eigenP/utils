
import numpy as np
import time
from scipy.ndimage import uniform_filter

def benchmark_aggregation():
    H, W = 2048, 2048
    patch_size = 128
    overlap = patch_size // 3

    # Random data
    data = np.random.rand(H, W).astype(np.float32)
    output = np.zeros_like(data)

    # 1. Uniform Filter
    start = time.time()
    for _ in range(20): # Simulate 20 slices
        uniform_filter(data, size=patch_size, output=output, mode='reflect')
    end = time.time()
    print(f"Uniform Filter time: {end - start:.4f}s")

    # 2. Integral Image (Cumsum)
    # We need to compute sums at specific grid points
    n_patches_y = H // (patch_size - overlap)
    n_patches_x = W // (patch_size - overlap)

    y_starts = np.arange(n_patches_y) * (patch_size - overlap)
    x_starts = np.arange(n_patches_x) * (patch_size - overlap)

    # Window coordinates
    # In uniform_filter, the value at (y, x) is sum over [y-P//2, y+P//2].
    # Integral image I(y, x) is sum of [0, y) x [0, x).
    # Sum of block [y1, y2) x [x1, x2) = I(y2, x2) - I(y1, x2) - I(y2, x1) + I(y1, x1).
    # We want window centered at yc, xc.
    # y1 = yc - P//2, y2 = yc + P//2 + 1?
    # Actually, uniform_filter size P means P pixels.
    # If P is even (128), origin is usually at P//2-1 or P//2.
    # Scipy uniform_filter centers the window.

    start = time.time()
    for _ in range(20):
        # Integral image (in-place on copy or reuse buffer)
        # We can reuse 'output' as buffer
        np.cumsum(data, axis=0, out=output)
        np.cumsum(output, axis=1, out=output)

        # Sampling (just simulating the overhead of gathering)
        # We would use integer indexing
        # For full comparison, we should extract the values
        # Let's assume we extract 500 points
        vals = output[0::100, 0::100]

    end = time.time()
    print(f"Integral Image time: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_aggregation()
