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

def estimate_shift_1d_iterative(ref_proj, moving_proj, window, max_iter=3, tol=0.01):
    """
    Estimates 1D shift using iterative windowed cross-correlation to eliminate stationary window bias.

    1. Estimate shift d.
    2. Shift moving signal by -d (aligning with ref).
    3. Re-estimate residual shift.
    4. Repeat until convergence.
    """
    current_shift = 0.0

    for _ in range(max_iter):
        shift_val = current_shift

        # Apply shift to align moving with ref
        # Using cubic interpolation (order=3) for accuracy
        if abs(shift_val) > 0.001:
            aligned_mov = shift(moving_proj, shift_val, order=3, mode='constant', cval=0)
        else:
            aligned_mov = moving_proj

        # Estimate residual shift
        # We multiply by window here. The window is stationary (matched to ref).
        # Since aligned_mov is aligned to ref, the window is correctly placed for both.
        res, _, _ = phase_cross_correlation(ref_proj * window, aligned_mov * window, upsample_factor=100)
        res = res[0]

        current_shift += res

        # Convergence check
        if abs(res) < tol:
            break

    return current_shift

def compute_drift_trajectory(
    projections_x: np.ndarray,
    projections_y: np.ndarray,
    mode: Literal['forward', 'reverse', 'both'] = 'forward',
    shape_limit: tuple = None,
    windows: tuple = None
) -> pd.DataFrame:
    """
    Computes the drift trajectory (cumulative corrections) for the video.

    Returns a DataFrame with columns: ['Time Point', 'dx', 'dy', 'cum_dx', 'cum_dy']
    where cum_dx/cum_dy are the shifts needed to align the frame to the reference.
    """
    T = projections_x.shape[0]

    # Initialize cumulative drift
    # We use an array to store cumulative drift for all timepoints
    cum_dx_arr = np.zeros(T, dtype=np.float32)
    cum_dy_arr = np.zeros(T, dtype=np.float32)

    # Steps
    dx_arr = np.zeros(T, dtype=np.float32)
    dy_arr = np.zeros(T, dtype=np.float32)

    win_x, win_y = windows if windows else (None, None)

    x_shape, y_shape = shape_limit if shape_limit else (1e9, 1e9) # Arbitrary large if None

    # Helper to clamp drift steps
    def clamp(val, limit):
        if abs(val) > limit:
            return 0.0
        return val

    if mode == 'reverse':
        # Reference is T-1
        # Iterate backwards: T-1 -> 1
        # We estimate shift to align t-1 to t

        for t in tqdm(range(T - 1, 0, -1), desc='Computing Drift Trajectory (Reverse)'):
            # t goes T-1, ..., 1. We are processing pair (t-1, t).
            # Target: Align t-1 to t.
            # estimate(ref, mov) returns shift to align mov to ref.
            # We want shift for t-1. So t-1 is moving. t is ref.

            # Using estimate_shift_1d_iterative(ref, mov)
            # Ref: t, Mov: t-1
            dx = estimate_shift_1d_iterative(projections_x[t], projections_x[t-1], win_x)
            dy = estimate_shift_1d_iterative(projections_y[t], projections_y[t-1], win_y)

            dx = clamp(dx, x_shape//5)
            dy = clamp(dy, y_shape//5)

            # P_{t-1} = P_t + dx
            # Since P is "correction vector", Correction(t-1) = Correction(t) + dx
            cum_dx_arr[t-1] = cum_dx_arr[t] + dx
            cum_dy_arr[t-1] = cum_dy_arr[t] + dy

            dx_arr[t-1] = dx
            dy_arr[t-1] = dy

    else:
        # Forward or Both
        # Reference is 0

        for t in tqdm(range(1, T), desc=f'Computing Drift Trajectory ({mode})'):
            # Align t to t-1

            if mode == 'both':
                # Bidirectional
                # dx_bwd: Align t to t-1 (Ref t-1, Mov t)
                dx_bwd = estimate_shift_1d_iterative(projections_x[t-1], projections_x[t], win_x)
                dy_bwd = estimate_shift_1d_iterative(projections_y[t-1], projections_y[t], win_y)

                # dx_fwd: Align t-1 to t (Ref t, Mov t-1)
                dx_fwd = estimate_shift_1d_iterative(projections_x[t], projections_x[t-1], win_x)
                dy_fwd = estimate_shift_1d_iterative(projections_y[t], projections_y[t-1], win_y)

                # Average step to align t to t-1
                # We want shift for t.
                # dx_bwd is shift for t.
                # dx_fwd is shift for t-1 (relative to t).
                # shift(t) approx -shift(t-1).
                # dx = (dx_bwd - dx_fwd) / 2
                dx = (dx_bwd - dx_fwd) / 2
                dy = (dy_bwd - dy_fwd) / 2

            else:
                # Forward only
                # Ref t-1, Mov t
                dx = estimate_shift_1d_iterative(projections_x[t-1], projections_x[t], win_x)
                dy = estimate_shift_1d_iterative(projections_y[t-1], projections_y[t], win_y)

            dx = clamp(dx, x_shape//5)
            dy = clamp(dy, y_shape//5)

            # Correction(t) = Correction(t-1) + dx
            cum_dx_arr[t] = cum_dx_arr[t-1] + dx
            cum_dy_arr[t] = cum_dy_arr[t-1] + dy

            dx_arr[t] = dx
            dy_arr[t] = dy

    # Create DataFrame
    df = pd.DataFrame({
        'Time Point': np.arange(T),
        'dx': dx_arr,
        'dy': dy_arr,
        'cum_dx': cum_dx_arr,
        'cum_dy': cum_dy_arr
    })

    return df


def apply_drift_correction_2D(
    video_data,
    reverse_time = False,
    save_drift_table=False,
    csv_filename='drift_table.csv',
    method: Literal['integer', 'subpixel'] = 'integer'
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
                   Note that 'subpixel' uses float shifts and performs range clipping to prevent integer wraparound artifacts.
    :return: A tuple containing two elements:
        - corrected_data: A 3D numpy array of the same shape as video_data, representing the drift-corrected video.
        - drift_table: A pandas DataFrame containing the drift values, cumulative drift, and time points.
    """
    # Get the dimensions of the video data
    t_shape, x_shape, y_shape = video_data.shape

    # Initialize an array to store the corrected video data
    corrected_data = np.zeros_like(video_data)

    # Initialize variables to store cumulative drift
    cum_dx, cum_dy = 0.0, 0.0

    # Initialize a list to store drift records for each time point
    drift_records = []


    min_value = video_data.min()

    # Pre-calculate data range limits for clipping (to prevent interpolation ringing)
    if np.issubdtype(video_data.dtype, np.integer):
        dtype_min = np.iinfo(video_data.dtype).min
        dtype_max = np.iinfo(video_data.dtype).max
    else:
        # For float data, we might not want to clip arbitrarily, or use the min/max of the data?
        # Usually float images are 0-1 or normalized.
        # Let's use the min/max of the data range if float, or just let it float.
        # However, to be consistent with 'zero_shift_multi_dimensional' which fills with fill_value,
        # we might want to respect bounds.
        # For now, we only clip integer types to prevent overflow.
        dtype_min, dtype_max = None, None

    # Pre-calculate weighted projections for all frames to avoid re-computation inside the loop
    # matth: Switched to decoupled windowing. No 2D weighting here.

    # Store projections: (T, W) and (T, H)
    projections_x = []
    projections_y = []

    # Iterate once to compute all raw projections
    for t in range(t_shape):
        projections_x.append(np.max(video_data[t], axis=0))
        projections_y.append(np.max(video_data[t], axis=1))

    # Convert to arrays
    # projections_x: (T, y_shape) -- Profile along X (Width)
    # projections_y: (T, x_shape) -- Profile along Y (Height)
    # Cast to float32 to allow windowing multiplication
    projections_x = np.array(projections_x, dtype=np.float32)
    projections_y = np.array(projections_y, dtype=np.float32)

    # Apply 1D Tukey window to the entire stack
    win_x = windows.tukey(y_shape, alpha=0.1).astype(np.float32)
    win_y = windows.tukey(x_shape, alpha=0.1).astype(np.float32)

    mode = 'forward'
    if reverse_time == 'both':
        mode = 'both'
    elif reverse_time:
        mode = 'reverse'

    # Compute trajectory first
    drift_table = compute_drift_trajectory(
        projections_x, projections_y,
        mode=mode,
        shape_limit=(x_shape, y_shape),
        windows=(win_x, win_y)
    )

    # Iterate all frames to apply correction
    for t in tqdm(range(t_shape), desc='Applying Drift Correction'):
        # Lookup drift
        # drift_table is indexed 0..T-1 and guaranteed to be sorted by Time Point by compute_drift_trajectory
        cum_dx = drift_table.loc[t, 'cum_dx']
        cum_dy = drift_table.loc[t, 'cum_dy']

        # Apply correction: shift by (cum_dy, cum_dx)
        # Note: input coords (y, x). shift=(dy, dx).

        # NOTE: We cast to integer for the shift operation, but keep cumulative drift as float
        # to prevent integrator windup/loss of precision for slow drifts.

        if method == 'subpixel':
            input_frame = video_data[t].astype(np.float32)

            shifted_slice = shift(
                input_frame,
                shift=(cum_dy, cum_dx),
                order=3,
                mode='constant',
                cval=min_value
            )

            if dtype_min is not None and dtype_max is not None:
                np.clip(shifted_slice, dtype_min, dtype_max, out=shifted_slice)

            corrected_data[t] = shifted_slice

        else:
            shift_dx = int(round(cum_dx))
            shift_dy = int(round(cum_dy))

            zero_shift_multi_dimensional(
                video_data[t],
                shifts=(shift_dy, shift_dx),
                fill_value=min_value,
                out=corrected_data[t]
            )

    # Optionally, save the drift table to a CSV file
    if save_drift_table:
        drift_table.to_csv(csv_filename, index=False)

    # Return the corrected video data and the drift table
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
