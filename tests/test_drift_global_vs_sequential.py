import numpy as np
import pytest
from scipy.ndimage import shift
from skimage import data, img_as_float
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

def test_random_walk_accumulation():
    """
    Demonstrates that sequential drift correction accumulates error (Random Walk).
    We generate a long sequence (T=50) with random drift.
    We check the residual position error after correction.
    """
    # 1. Setup Data
    T = 50
    H, W = 128, 128

    # Base image: Sparse Gaussian Spot (simulating fluorescence)
    # This avoids windowing bias artifacts present in dense images (like camera)
    # when drift is large.
    y_grid, x_grid = np.indices((H, W))
    cy, cx = H//2, W//2
    sigma = 5
    img = np.exp(-((y_grid-cy)**2 + (x_grid-cx)**2) / (2*sigma**2))
    img = img.astype(np.float32)

    # Generate random walk trajectory
    np.random.seed(42)
    # Drift steps: Normal(0, 0.5)
    # Use slightly larger drift to ensure we have signal to track
    drift_steps = np.random.normal(0, 0.5, size=(T, 2))
    drift_steps[0] = 0 # No drift at t=0

    # True trajectory (cumulative)
    true_trajectory = np.cumsum(drift_steps, axis=0)

    # Create video
    video = np.zeros((T, H, W), dtype=np.float32)

    # Create a window to avoid edge effects which might confuse simple shift tests
    # But apply_drift_correction_2D handles edges via windowing, so standard crop is fine.
    # We use mode='reflect' to simulate continuous scene.

    for t in range(T):
        # Shift image by true trajectory
        shift_val = true_trajectory[t]
        video[t] = shift(img, shift=shift_val, order=1, mode='reflect')

    # 2. Apply Drift Correction (Sequential - Default)
    # Use subpixel method for best accuracy
    corrected, table = apply_drift_correction_2D(video, method='subpixel')

    # 3. Analyze Residuals
    # The estimated trajectory (correction) should be -True_Trajectory
    # The table misses T=0 (assumed 0 drift relative to itself)
    # We insert 0 at the beginning
    estimated_traj_y = np.concatenate(([0], table['cum_dy'].values))
    estimated_traj_x = np.concatenate(([0], table['cum_dx'].values))

    # Residual = True + Estimated (should be 0)
    diff_y = true_trajectory[:, 0] + estimated_traj_y
    diff_x = true_trajectory[:, 1] + estimated_traj_x

    residuals = np.sqrt(diff_y**2 + diff_x**2)

    rmse = np.sqrt(np.mean(residuals**2))
    max_err = np.max(residuals)
    final_err = residuals[-1]

    print(f"\nSequential Baseline Results (T={T}):")
    print(f"RMSE: {rmse:.4f} px")
    print(f"Max Error: {max_err:.4f} px")
    print(f"Final Drift Error: {final_err:.4f} px")

    # Verify that 'global' mode works (if implemented) or fails if not
    # 4. Apply Drift Correction (Global)
    print("\nRunning Global Refinement...")
    corrected_global, table_global = apply_drift_correction_2D(video, method='subpixel', mode='global')

    # Analyze Global Residuals
    # Note: Global mode table usually contains all frames, but logic filters frame 0 if not reverse.
    # We should check the table length.
    if len(table_global) == T - 1:
        estimated_traj_y_g = np.concatenate(([0], table_global['cum_dy'].values))
        estimated_traj_x_g = np.concatenate(([0], table_global['cum_dx'].values))
    else:
        # If I updated it to return all frames, logic differs.
        # Based on my implementation: "if not reverse_time: drift_table = drift_table[drift_table['Time Point'] > 0]"
        # So it returns T-1 frames.
        estimated_traj_y_g = np.concatenate(([0], table_global['cum_dy'].values))
        estimated_traj_x_g = np.concatenate(([0], table_global['cum_dx'].values))

    diff_y_g = true_trajectory[:, 0] + estimated_traj_y_g
    diff_x_g = true_trajectory[:, 1] + estimated_traj_x_g

    residuals_g = np.sqrt(diff_y_g**2 + diff_x_g**2)

    rmse_g = np.sqrt(np.mean(residuals_g**2))
    max_err_g = np.max(residuals_g)
    final_err_g = residuals_g[-1]

    print(f"\nGlobal Results (T={T}):")
    print(f"RMSE: {rmse_g:.4f} px")
    print(f"Max Error: {max_err_g:.4f} px")
    print(f"Final Drift Error: {final_err_g:.4f} px")

    # Assert improvement
    # We expect significant improvement (e.g. > 50% reduction in RMSE)
    # But since noise is high (0.5 px), maybe we should be conservative.
    # Sequential error accumulates. Global error should be ~ noise.
    # With T=50, Sequential error ~ 3px. Global error ~ 0.5px.
    assert rmse_g < rmse * 0.75, f"Global RMSE ({rmse_g}) should be lower than Sequential ({rmse})"

    return residuals, residuals_g

if __name__ == "__main__":
    test_random_walk_accumulation()
