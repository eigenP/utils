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
import logging
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

def estimate_drift(frame1, frame2, return_ccm = False):
    """
    Estimate the xy-drift between two 2D frames using cross-correlation.
    NOTE: Currently 2D only. For 3D+t data, consider projecting or using the full pipeline.

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
    projections: list,
    mode: Literal['forward', 'reverse', 'both'] = 'forward',
    shape_limit: tuple = None,
    windows: tuple = None
) -> pd.DataFrame:
    """
    Computes the drift trajectory (cumulative corrections) for the video.

    :param projections: List of numpy arrays, one for each spatial dimension (e.g., [proj_z, proj_y, proj_x]).
    :param mode: 'forward', 'reverse', or 'both'
    :param shape_limit: Tuple of maximum shifts to clamp (e.g., (z_shape, y_shape, x_shape))
    :param windows: Tuple of 1D windows to apply to each projection (e.g., (win_z, win_y, win_x))
    Returns a DataFrame with columns: ['Time Point', 'dx', 'dy', 'dz', 'cum_dx', 'cum_dy', 'cum_dz']
    where cum_d* are the shifts needed to align the frame to the reference.
    If 2 spatial dimensions are provided, dz and cum_dz will be 0.
    """
    num_dims = len(projections)
    T = projections[0].shape[0]

    # Initialize steps and cumulative drift for 3 axes (z, y, x)
    # If 2D (num_dims == 2), we will just map index 0 -> y, index 1 -> x
    # If 3D (num_dims == 3), index 0 -> z, index 1 -> y, index 2 -> x
    cum_d = np.zeros((3, T), dtype=np.float32)
    d = np.zeros((3, T), dtype=np.float32)

    wins = windows if windows else [None] * num_dims
    limits = shape_limit if shape_limit else [1e9] * num_dims

    # Helper to clamp drift steps
    def clamp(val, limit):
        if abs(val) > limit:
            return 0.0
        return val

    # Helper to process a single time step align `mov_idx` to `ref_idx`
    def align_step(ref_idx, mov_idx, is_bwd=False):
        shifts = []
        for d_idx in range(num_dims):
            shift_val = estimate_shift_1d_iterative(
                projections[d_idx][ref_idx],
                projections[d_idx][mov_idx],
                wins[d_idx]
            )
            shifts.append(shift_val)
        return shifts

    if mode == 'reverse':
        # Iterate backwards: T-1 -> 1
        for t in tqdm(range(T - 1, 0, -1), desc='Computing Drift Trajectory (Reverse)'):
            shifts = align_step(t, t-1)
            for d_idx in range(num_dims):
                c = clamp(shifts[d_idx], limits[d_idx] // 5)
                # target axis in output arrays:
                # If 2D (y, x), d_idx=0 -> y (axis 1), d_idx=1 -> x (axis 2)
                # If 3D (z, y, x), d_idx=0 -> z (axis 0), d_idx=1 -> y (axis 1), d_idx=2 -> x (axis 2)
                out_idx = d_idx + (3 - num_dims)
                d[out_idx, t-1] = c
                cum_d[out_idx, t-1] = cum_d[out_idx, t] + c

    else:
        # Forward or Both
        for t in tqdm(range(1, T), desc=f'Computing Drift Trajectory ({mode})'):
            if mode == 'both':
                shifts_bwd = align_step(t-1, t)
                shifts_fwd = align_step(t, t-1)
                shifts = [(b - f) / 2 for b, f in zip(shifts_bwd, shifts_fwd)]
            else:
                shifts = align_step(t-1, t)

            for d_idx in range(num_dims):
                c = clamp(shifts[d_idx], limits[d_idx] // 5)
                out_idx = d_idx + (3 - num_dims)
                d[out_idx, t] = c
                cum_d[out_idx, t] = cum_d[out_idx, t-1] + c

    # Create DataFrame
    df = pd.DataFrame({
        'Time Point': np.arange(T),
        'dx': d[2],
        'dy': d[1],
        'dz': d[0],
        'cum_dx': cum_d[2],
        'cum_dy': cum_d[1],
        'cum_dz': cum_d[0]
    })

    return df


def apply_drift_correction(
    video_data,
    reverse_time = False,
    save_drift_table=False,
    csv_filename='drift_table.csv',
    method: Literal['integer', 'subpixel'] = 'integer'
):
    """
    Apply drift correction to video data (2D+t or 3D+t).

    This function corrects for drift in video data frame by frame. It projects the data
    along spatial dimensions to estimate 1D shifts, and applies corrections to align the frames.

    :param video_data: A 3D (T, Y, X) or 4D (T, Z, Y, X) numpy array representing the video data.
    :param reverse_time: Process frames in reverse order (or 'both').
    :param save_drift_table: A boolean indicating whether to save the drift values to a CSV file. Default is False.
    :param csv_filename: The name of the CSV file to save the drift table to. Default is 'drift_table.csv'.
    :param method: 'integer' (default) for fast integer shifting, or 'subpixel' for precise bicubic interpolation.
    :return: A tuple containing two elements:
        - corrected_data: A numpy array of the same shape as video_data.
        - drift_table: A pandas DataFrame containing the drift values.
    """
    ndim = video_data.ndim
    if ndim == 3:
        logging.info("Input is 3D (T, Y, X). Proceeding with 2D+t max-projection registration.")
        t_shape, y_shape, x_shape = video_data.shape
        num_spatial_dims = 2
    elif ndim == 4:
        logging.info("Input is 4D (T, Z, Y, X). Proceeding with 3D+t max-projection registration.")
        t_shape, z_shape, y_shape, x_shape = video_data.shape
        num_spatial_dims = 3
    else:
        raise ValueError(f"Expected 3D (T, Y, X) or 4D (T, Z, Y, X) data, got {ndim}D data.")

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

    projections = []
    shape_limit = []
    windows_tuple = []

    if num_spatial_dims == 2:
        # 2D+t -> Y, X
        proj_y = []
        proj_x = []
        for t in range(t_shape):
            # Axis 0 in video_data[t] is Y, Axis 1 is X
            proj_y.append(np.max(video_data[t], axis=1)) # project X, gives Y profile
            proj_x.append(np.max(video_data[t], axis=0)) # project Y, gives X profile

        proj_y = np.array(proj_y, dtype=np.float32)
        proj_x = np.array(proj_x, dtype=np.float32)

        projections = [proj_y, proj_x]
        shape_limit = (y_shape, x_shape)
        windows_tuple = (
            windows.tukey(y_shape, alpha=0.1).astype(np.float32),
            windows.tukey(x_shape, alpha=0.1).astype(np.float32)
        )
    else:
        # 3D+t -> Z, Y, X
        proj_z = []
        proj_y = []
        proj_x = []
        for t in range(t_shape):
            # Axis 0 is Z, Axis 1 is Y, Axis 2 is X in video_data[t]
            proj_z.append(np.max(video_data[t], axis=(1, 2))) # project Y,X -> Z profile
            proj_y.append(np.max(video_data[t], axis=(0, 2))) # project Z,X -> Y profile
            proj_x.append(np.max(video_data[t], axis=(0, 1))) # project Z,Y -> X profile

        proj_z = np.array(proj_z, dtype=np.float32)
        proj_y = np.array(proj_y, dtype=np.float32)
        proj_x = np.array(proj_x, dtype=np.float32)

        projections = [proj_z, proj_y, proj_x]
        shape_limit = (z_shape, y_shape, x_shape)
        windows_tuple = (
            windows.tukey(z_shape, alpha=0.1).astype(np.float32),
            windows.tukey(y_shape, alpha=0.1).astype(np.float32),
            windows.tukey(x_shape, alpha=0.1).astype(np.float32)
        )

    mode = 'forward'
    if reverse_time == 'both':
        mode = 'both'
    elif reverse_time:
        mode = 'reverse'

    # Compute trajectory first
    drift_table = compute_drift_trajectory(
        projections,
        mode=mode,
        shape_limit=shape_limit,
        windows=windows_tuple
    )

    # Iterate all frames to apply correction
    for t in tqdm(range(t_shape), desc='Applying Drift Correction'):
        # Lookup drift
        # drift_table is indexed 0..T-1 and guaranteed to be sorted by Time Point by compute_drift_trajectory
        cum_dx = drift_table.loc[t, 'cum_dx']
        cum_dy = drift_table.loc[t, 'cum_dy']
        cum_dz = drift_table.loc[t, 'cum_dz']

        # NOTE: We cast to integer for the shift operation, but keep cumulative drift as float
        # to prevent integrator windup/loss of precision for slow drifts.

        if num_spatial_dims == 2:
            shift_tuple = (cum_dy, cum_dx)
            int_shift_tuple = (int(round(cum_dy)), int(round(cum_dx)))
        else:
            shift_tuple = (cum_dz, cum_dy, cum_dx)
            int_shift_tuple = (int(round(cum_dz)), int(round(cum_dy)), int(round(cum_dx)))

        if method == 'subpixel':
            input_frame = video_data[t].astype(np.float32)

            shifted_slice = shift(
                input_frame,
                shift=shift_tuple,
                order=3,
                mode='constant',
                cval=min_value
            )

            if dtype_min is not None and dtype_max is not None:
                np.clip(shifted_slice, dtype_min, dtype_max, out=shifted_slice)

            corrected_data[t] = shifted_slice

        else:
            zero_shift_multi_dimensional(
                video_data[t],
                shifts=int_shift_tuple,
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
