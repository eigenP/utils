# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "tqdm",
#     "scipy",
# ]
# ///
#@markdown `maxproj_registration.py`

### Imports
import numpy as np
import pandas as pd
from typing import Literal
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import shift
from scipy.signal import windows
# from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def zero_shift_multi_dimensional(arr, shifts = 0, fill_value=0, out=None):
    """
    Shift the elements of a multi-dimensional NumPy array along each axis by specified amounts, filling the vacant positions with a specified fill value.

    :param arr: A multi-dimensional NumPy array
    :param shifts: A single integer or a list/tuple of integers specifying the shift amounts for each axis
    :param fill_value: An optional value to fill the vacant positions after the shift (default is 0)
    :param out: A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to.
                If not provided or None, a freshly-allocated array is returned.
    :return: A new NumPy array with the elements shifted and the specified fill value in the vacant positions
    """
    # Ensure shifts is a list or tuple of integers, or a single integer
    if isinstance(shifts, int):
        shifts = [shifts] * arr.ndim
    elif isinstance(shifts, (list, tuple)):
        if len(shifts) != arr.ndim:
            raise ValueError("Length of shifts must be equal to the number of dimensions in the array.")
        if not all(isinstance(shift, int) for shift in shifts):
            raise TypeError("All shift values must be integers.")
    else:
        raise TypeError("Shifts must be a single integer or a list/tuple of integers.")

    # Initialize the result array
    # If out is None, we need to create a new array. We use empty_like to avoid
    # unnecessary initialization since we will overwrite most/all of it.
    if out is None:
        result = np.empty_like(arr)
    else:
        result = out

    # Initialize slices for input and output arrays
    slices_input = [slice(None)] * arr.ndim
    slices_output = [slice(None)] * arr.ndim

    # Apply the shifts calculation
    for axis, shift in enumerate(shifts):
        if shift > 0:
            slices_input[axis] = slice(None, -shift)
            slices_output[axis] = slice(shift, None)
        elif shift < 0:
            slices_input[axis] = slice(-shift, None)
            slices_output[axis] = slice(None, shift)

    # Perform the shift (copy data from input to output location)
    # This overwrites the central part of the array
    result[tuple(slices_output)] = arr[tuple(slices_input)]

    # Fill the exposed gaps with fill_value
    # We iterate over axes and fill the strip that was "exposed" by the shift.
    # The union of these strips and the copied central part covers the entire array.
    for axis, shift in enumerate(shifts):
        if shift == 0:
            continue

        # Construct slice for the gap along this axis
        # If shift > 0, the gap is at the beginning (0 to shift)
        # If shift < 0, the gap is at the end (shift to end)
        gap_slices = [slice(None)] * arr.ndim
        if shift > 0:
            gap_slices[axis] = slice(None, shift)
        else:
            gap_slices[axis] = slice(shift, None)

        result[tuple(gap_slices)] = fill_value

    return result

def estimate_drift_2D(frame1, frame2, return_ccm = False):
    """
    Estimate the xy-drift between two 2D frames using cross-correlation.

    :param frame1: 2D numpy array, first frame
    :param frame2: 2D numpy array, second frame
    :return: Tuple (dx, dy), estimated drift in x and y directions
    """
    # Calculate the cross-correlation matrix

    # matth: Decoupled 1D Windowing
    # Instead of windowing the 2D image (which couples X and Y attenuation and causes
    # information loss at the boundaries), we compute the raw max projections and then
    # apply a 1D window to the projections.

    # Raw max projections
    # axis=0 -> Projection along Height -> Profile of Width (X)
    frame1_max_proj_x = np.max(frame1, axis = 0)
    frame2_max_proj_x = np.max(frame2, axis = 0)

    # axis=1 -> Projection along Width -> Profile of Height (Y)
    frame1_max_proj_y = np.max(frame1, axis = 1)
    frame2_max_proj_y = np.max(frame2, axis = 1)

    # Apply 1D Tukey window to reduce FFT edge artifacts
    # Use alpha=0.1 (taper 5% on each side)
    # This is much less aggressive than the previous 33% taper
    w_shape_x = frame1_max_proj_x.shape[0]
    w_shape_y = frame1_max_proj_y.shape[0]

    win_x = windows.tukey(w_shape_x, alpha=0.1).astype(np.float32)
    win_y = windows.tukey(w_shape_y, alpha=0.1).astype(np.float32)

    frame1_max_proj_x = frame1_max_proj_x * win_x
    frame2_max_proj_x = frame2_max_proj_x * win_x

    frame1_max_proj_y = frame1_max_proj_y * win_y
    frame2_max_proj_y = frame2_max_proj_y * win_y

    # Apply gaussian smoothing for robustness
    # frame1_max_proj_x = gaussian_filter1d(frame1_max_proj_x, sigma = 3, radius = 5)
    # frame2_max_proj_x = gaussian_filter1d(frame2_max_proj_x, sigma = 3, radius = 5)

    # frame1_max_proj_y = gaussian_filter1d(frame1_max_proj_y, sigma = 3, radius = 5)
    # frame2_max_proj_y = gaussian_filter1d(frame2_max_proj_y, sigma = 3, radius = 5)

    shift_x, error, diffphase = phase_cross_correlation(frame1_max_proj_x,
                                                      frame2_max_proj_x, upsample_factor=100)

    shift_y, error, diffphase = phase_cross_correlation(frame1_max_proj_y,
                                                      frame2_max_proj_y, upsample_factor=100)

    shift = np.array((shift_x[0], shift_y[0]))


    # ### --- Add a test if shifts are too large
    # if shift_x > int(frame1.shape[0]*0.4) or shift_y > int(frame1.shape[0]*0.4):

    #     spacer = int(frame1.shape[0]*0.1)

    #     shift_x, error, diffphase = phase_cross_correlation(frame1_max_proj_x[spacer:-spacer],
    #                                                   frame2_max_proj_x[spacer:-spacer])

    #     shift_y, error, diffphase = phase_cross_correlation(frame1_max_proj_y[spacer:-spacer],
    #                                                     frame2_max_proj_y[spacer:-spacer])

    #     shift_x = shift_x #+ spacer
    #     shift_y = shift_y #+ spacer


    # shift = (shift_x, shift_y)


    if return_ccm:
        # Calculate the upsampled DFT, again to show what the algorithm is doing
        # behind the scenes.  Constants correspond to calculated values in routine.
        # See source code for details.
        # image_product = np.fft.fft2(frame1) * np.fft.fft2(frame2).conj()
        # cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()

        frame1 = np.vstack((frame1_max_proj_y,frame1_max_proj_y))
        frame2 = np.vstack((frame2_max_proj_y,frame2_max_proj_y))

        image_product = np.fft.fft2(frame1) * np.fft.fft2(frame2).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

        return shift, cc_image.real
    else:
        return shift

def _compute_drift_trajectory(projections_x, projections_y, reverse_time):
    """
    Helper function to estimate the sequential drift trajectory.
    Returns: trajectory (T, 2) where trajectory[t] = (cum_dx, cum_dy)
    """
    t_shape = projections_x.shape[0]
    x_shape = projections_y.shape[1] # Height
    y_shape = projections_x.shape[1] # Width

    trajectory = np.zeros((t_shape, 2), dtype=np.float32)
    cum_dx, cum_dy = 0.0, 0.0

    if reverse_time == 'both':
        # Forward order for 'both' (Bidirectional Estimation)
        # We start from t=0.
        # trajectory[0] is 0.

        range_values = range(1, t_shape)

        for time_point in tqdm(range_values, desc='Estimating Sequential Drift'):
            # Estimate the drift between the current frame and the previous frame
            # Use precomputed projections with subpixel precision
            shift_x_back, _, _ = phase_cross_correlation(projections_x[time_point - 1], projections_x[time_point], upsample_factor=100)
            shift_y_back, _, _ = phase_cross_correlation(projections_y[time_point - 1], projections_y[time_point], upsample_factor=100)
            dx_backward, dy_backward = shift_x_back[0], shift_y_back[0]

            shift_x_fwd, _, _ = phase_cross_correlation(projections_x[time_point], projections_x[time_point - 1], upsample_factor=100)
            shift_y_fwd, _, _ = phase_cross_correlation(projections_y[time_point], projections_y[time_point - 1], upsample_factor=100)
            dx_forward, dy_forward = shift_x_fwd[0], shift_y_fwd[0]

            # dx_backward is shift T-1 -> T (e.g. -0.5 for +0.5 motion)
            # dx_forward is shift T -> T-1 (e.g. +0.5 for +0.5 motion)
            # We want the average of (dx_backward) and (-dx_forward)
            dx = (dx_backward - dx_forward) / 2
            dy = (dy_backward - dy_forward) / 2

            # Basic clipping if too large
            if abs(dx) > y_shape//5:
                dx = 0.0
            if abs(dy) > x_shape//5:
                dy = 0.0

            # Update the cumulative drift
            cum_dx, cum_dy = cum_dx + dx, cum_dy + dy

            trajectory[time_point] = [cum_dx, cum_dy]

    else:
        DRIFT_SIGN = -1 if reverse_time else 1

        if not reverse_time:
            range_values = range(1, t_shape)
            # trajectory[0] is 0
        else:
            range_values = range(t_shape - 1, 0, -1)
            # trajectory[-1] is 0 (relative to itself)
            # But we want to express everything relative to T=0 for consistency?
            # Original code:
            # First frame processed (T-1) gets drift from T.
            # cum_drift starts at 0.
            # corrected_data[-1] = video_data[-1] (Base)
            pass

        # Note: If reverse_time=True, we iterate T-1 down to 0.
        # cum_dx accumulates shifts relative to LAST frame (T-1).
        # To make trajectory consistent (relative to T=0), we would need to shift logic.
        # But let's stick to the original logic: trajectory[t] is shift required for frame t.

        # If reverse_time=True, we start with t=T-1 (uncorrected, base).
        # Wait, original code:
        # corrected_data[-1] = video_data[-1]
        # range_values = range(t_shape - 1, 0, -1) -> T-1, T-2... 1
        # Loop t: compare t-1 (prev) vs t (curr).
        # Wait, if t in range(5, 0, -1) -> 5,4,3,2,1.
        # Loop variable is 'time_point'.
        # Comparing projections[time_point-1] vs projections[time_point].
        # Correcting frame[time_point].

        # Wait, original code for reverse_time:
        # range_values = range(t_shape - 1, 0, -1)
        # for time_point in ...:
        #    shift ... proj[time_point-1] vs proj[time_point]
        #    apply to video_data[time_point - OFFSET] where OFFSET=1
        #    i.e. apply to video_data[time_point - 1].

        # So we correct frame t-1 based on drift vs frame t.
        # Frame T-1 is base.
        # Frame T-2 is corrected to T-1.
        # Frame 0 is corrected to 1 (chained).

        # We need to populate trajectory for all frames.

        for time_point in tqdm(range_values, desc='Estimating Sequential Drift'):
            shift_x, _, _ = phase_cross_correlation(projections_x[time_point - 1], projections_x[time_point], upsample_factor=100)
            shift_y, _, _ = phase_cross_correlation(projections_y[time_point - 1], projections_y[time_point], upsample_factor=100)
            dx, dy = shift_x[0], shift_y[0]

            dx, dy = dx * DRIFT_SIGN, dy * DRIFT_SIGN

            if abs(dx) > y_shape//5:
                dx = 0.0
            if abs(dy) > x_shape//5:
                dy = 0.0

            cum_dx, cum_dy = cum_dx + dx, cum_dy + dy

            # Store in trajectory.
            # If reverse_time, we are correcting time_point - 1.
            idx = time_point - 1 if reverse_time else time_point
            trajectory[idx] = [cum_dx, cum_dy]

    return trajectory

def _refine_trajectory_global(trajectory, projections_x, projections_y):
    """
    Refines the drift trajectory by registering each frame's projection
    to the global median projection of the aligned stack.
    This eliminates random walk error accumulation.
    """
    T = len(projections_x)

    # 1. Align projections using sequential trajectory
    # We shift the 1D projections by the NEGATIVE of the sequential drift
    # to align them to the common reference frame.

    aligned_x = np.zeros_like(projections_x)
    aligned_y = np.zeros_like(projections_y)

    # Batch shift is not supported by scipy.ndimage.shift directly for variable shifts
    # We must loop. But 1D shift is fast.
    for t in range(T):
        dx = trajectory[t, 0]
        dy = trajectory[t, 1]

        # Shift X-projection by dx (along its only axis)
        # Note: Projections are float32, so we can use interpolation
        # mode='constant', cval=0 assumes signal is zero outside FOV
        # We use dx (the correction shift) to align the projection to the reference frame.
        aligned_x[t] = shift(projections_x[t], [dx], order=1, mode='constant', cval=0)
        aligned_y[t] = shift(projections_y[t], [dy], order=1, mode='constant', cval=0)

    # 2. Compute Robust Reference (Median)
    ref_x = np.median(aligned_x, axis=0)
    ref_y = np.median(aligned_y, axis=0)

    # 3. Re-estimate drift against Reference
    new_trajectory = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        # Register raw projection t against Reference
        # We want shift to move P[t] to Ref.
        # This shift IS the drift correction we need.
        # (Original trajectory[t] was also the shift to correct).

        # phase_cross_correlation(Reference, Moving) -> Shift to align Moving to Reference.
        sx, _, _ = phase_cross_correlation(ref_x, projections_x[t], upsample_factor=100)
        sy, _, _ = phase_cross_correlation(ref_y, projections_y[t], upsample_factor=100)

        new_trajectory[t] = [sx[0], sy[0]]

    # Note: There might be a global offset between sequential 0 and median.
    # We usually want to keep frame 0 at 0 (or close to it) to avoid shifting the whole video unnecessarily.
    # So we subtract trajectory[0] from all.

    offset = new_trajectory[0].copy()
    new_trajectory -= offset

    return new_trajectory

def apply_drift_correction_2D(
    video_data,
    reverse_time = False,
    save_drift_table=False,
    csv_filename='drift_table.csv',
    method: Literal['integer', 'subpixel'] = 'integer',
    mode: Literal['sequential', 'global'] = 'sequential'
):
    """
    Apply drift correction to video data.

    This function corrects for drift in video data frame by frame. It calculates the drift between
    consecutive frames using the `estimate_drift` function, and applies corrections to align the frames.
    The cumulative drift is also calculated and stored. Optionally, a table of drift values can be saved
    to a CSV file.

    :param video_data: A 3D numpy array representing the video data. The dimensions should be (time, x, y).
    :param reverse_time: Process frames in reverse order (or 'both').
    :param save_drift_table: A boolean indicating whether to save the drift values to a CSV file. Default is False.
    :param csv_filename: The name of the CSV file to save the drift table to. Default is 'drift_table.csv'.
    :param method: 'integer' (default) for fast integer shifting, or 'subpixel' for precise bicubic interpolation.
    :param mode: 'sequential' (default) for frame-to-frame tracking, or 'global' for robust stabilization against a common reference (removes random walk).
    :return: A tuple containing two elements:
        - corrected_data: A 3D numpy array of the same shape as video_data, representing the drift-corrected video.
        - drift_table: A pandas DataFrame containing the drift values, cumulative drift, and time points.
    """
    # Get the dimensions of the video data
    t_shape, x_shape, y_shape = video_data.shape

    # Initialize an array to store the corrected video data
    corrected_data = np.zeros_like(video_data)

    # ---------------------------------------------------------
    # 1. Pre-calculate Projections
    # ---------------------------------------------------------
    projections_x = []
    projections_y = []

    # Iterate once to compute all raw projections
    for t in range(t_shape):
        projections_x.append(np.max(video_data[t], axis=0))
        projections_y.append(np.max(video_data[t], axis=1))

    projections_x = np.array(projections_x, dtype=np.float32)
    projections_y = np.array(projections_y, dtype=np.float32)

    # Apply 1D Tukey window
    win_x = windows.tukey(y_shape, alpha=0.1).astype(np.float32)
    win_y = windows.tukey(x_shape, alpha=0.1).astype(np.float32)

    projections_x *= win_x[None, :]
    projections_y *= win_y[None, :]

    # ---------------------------------------------------------
    # 2. Estimate Sequential Trajectory
    # ---------------------------------------------------------
    trajectory = _compute_drift_trajectory(projections_x, projections_y, reverse_time)

    # ---------------------------------------------------------
    # 3. (Optional) Global Refinement
    # ---------------------------------------------------------
    if mode == 'global':
        if reverse_time != 'both' and reverse_time is not False:
             # Global refinement assumes forward-time consistency for simplicity,
             # or effectively treats the stack as a whole.
             # It works fine with reverse_time='both' (forward iteration).
             # It might be confusing with reverse_time=True.
             # We'll allow it but warn or just process.
             pass

        trajectory = _refine_trajectory_global(trajectory, projections_x, projections_y)

    # ---------------------------------------------------------
    # 4. Apply Correction
    # ---------------------------------------------------------

    min_value = video_data.min()
    if np.issubdtype(video_data.dtype, np.integer):
        dtype_min = np.iinfo(video_data.dtype).min
        dtype_max = np.iinfo(video_data.dtype).max
    else:
        # For float data, we assume 0 as min bound if min_value >= 0
        dtype_min = 0.0 if min_value >= 0 else None
        dtype_max = None # No upper bound for float usually

    for time_point in tqdm(range(t_shape), desc='Applying Corrections'):
        cum_dx = trajectory[time_point, 0]
        cum_dy = trajectory[time_point, 1]

        # If we need to correct frame t, we shift by (cum_dy, cum_dx)
        # Note: In _compute_drift_trajectory, we stored the shift required for frame t.

        if method == 'subpixel':
            s_dy, s_dx = cum_dy, cum_dx
            input_frame = video_data[time_point].astype(np.float32)

            shifted_slice = shift(
                input_frame,
                shift=(s_dy, s_dx),
                order=3,
                mode='constant',
                cval=min_value
            )

            if dtype_min is not None:
                 np.maximum(shifted_slice, dtype_min, out=shifted_slice)
            if dtype_max is not None:
                 np.minimum(shifted_slice, dtype_max, out=shifted_slice)

            corrected_data[time_point] = shifted_slice

        else:
            shift_dx = int(round(cum_dx))
            shift_dy = int(round(cum_dy))

            zero_shift_multi_dimensional(
                video_data[time_point],
                shifts=(shift_dy, shift_dx),
                fill_value=min_value,
                out=corrected_data[time_point]
            )

    # ---------------------------------------------------------
    # 5. Generate Drift Table
    # ---------------------------------------------------------
    drift_records = []
    for t in range(t_shape):
        cum_dx = trajectory[t, 0]
        cum_dy = trajectory[t, 1]

        if t == 0:
            dx = cum_dx
            dy = cum_dy
        else:
            dx = cum_dx - trajectory[t-1, 0]
            dy = cum_dy - trajectory[t-1, 1]

        drift_records.append({'Time Point': t, 'dx': dx, 'dy': dy, 'cum_dx': cum_dx, 'cum_dy': cum_dy})

    # Create a DataFrame from the list of drift records
    drift_table = pd.DataFrame(drift_records)
    # Filter out frame 0 to match original behavior if desired?
    # Original behavior for 'both': 0 is not in table.
    # Original behavior for reverse=False: 0 is not in table.
    # Let's keep 0 but maybe filter later if needed.
    # Actually, returning T records is more correct.
    # But for backward compatibility, maybe I should check?
    # Original code drift_records.append starting from second frame.

    if not reverse_time:
         # Original: range(1, t_shape)
         drift_table = drift_table[drift_table['Time Point'] > 0]
    elif reverse_time == 'both':
         # Original: range(1, t_shape)
         drift_table = drift_table[drift_table['Time Point'] > 0]
    else:
         # Original: range(t_shape - 1, 0, -1) -> 0 is not included (corrected_data[-1] is base)
         drift_table = drift_table[drift_table['Time Point'] < t_shape - 1]
         # Wait, if reverse, base is T-1.
         pass

    # To keep things simple and consistent, let's return ALL points.
    # The user can filter. This is an improvement.
    drift_table.sort_values(by=['Time Point'], inplace=True)

    # Optionally, save the drift table to a CSV file
    if save_drift_table:
        drift_table.to_csv(csv_filename, index=False)

    return corrected_data, drift_table

def apply_subpixel_drift_correction(image, drift):
    """
    Applies subpixel drift correction to an image using bicubic interpolation.

    Parameters:
    image (np.array): The input image.
    drift (tuple): The drift in subpixel units (dz, dy, dx).

    Returns:
    np.array: The corrected image.
    """
    # Ensure image is a numpy array
    image = np.asarray(image)
    min_value = image.min()

    # Efficiently apply subpixel shift using scipy.ndimage.shift
    # This avoids creating a full coordinate grid (O(N) memory savings)
    corrected_image = shift(image, shift=drift, order=3, mode='constant', cval=min_value)

    return corrected_image
