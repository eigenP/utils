
import numpy as np
import time
import cProfile
import pstats
from eigenp_utils.extended_depth_of_focus import best_focus_image

def benchmark():
    # Large 3D stack: 20 slices, 2048x2048
    Z, H, W = 20, 2048, 2048
    print(f"Creating synthetic stack ({Z}, {H}, {W})...")

    img = np.random.rand(Z, H, W).astype(np.float32)

    print("Running best_focus_image...")

    profiler = cProfile.Profile()
    profiler.enable()

    result = best_focus_image(img, patch_size=128)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)

if __name__ == "__main__":
    benchmark()
