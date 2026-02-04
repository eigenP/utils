# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "tqdm",
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

def _get_weight_profiles(shape, overlap):
    """
    Generate 1D weight profiles for Y and X axes.
    """
    H, W = shape

    if overlap <= 0:
        return np.ones(H, dtype=np.float32), np.ones(W, dtype=np.float32)

    # 1D weight function based on cubic spline
    def weight_1d(x):
        return 3 * x**2 - 2 * x**3

    # Generate the 1D taper profile
    # Using float32 for better precision than float16, avoiding accumulation errors
    x = np.linspace(0, 1, overlap)
    taper = weight_1d(x).astype(np.float32)

    # Construct 1D profiles for Y and X axes
    # The profile is 1.0 in the center and tapers to 0.0 at the edges
    profile_y = np.ones(H, dtype=np.float32)
    profile_y[:overlap] = taper
    profile_y[-overlap:] = taper[::-1]

    profile_x = np.ones(W, dtype=np.float32)
    profile_x[:overlap] = taper
    profile_x[-overlap:] = taper[::-1]

    return profile_y, profile_x


def _2D_weighted_image(image, overlap, profiles=None, out=None):
    '''
    # image := image shape
    # overlap := in pixels
    # profiles := optional pre-calculated (profile_y, profile_x)
    # out := optional output array for in-place operation

    # Example usage
    # _2D_window = _2D_weight(image, overlap)
    '''
    if overlap <= 0:
        if out is not None:
            out[:] = image
            return out
        return image.astype(np.float32)

    if profiles is None:
        profile_y, profile_x = _get_weight_profiles(image.shape, overlap)
    else:
        profile_y, profile_x = profiles

    if out is None:
        # Apply weights using broadcasting
        # (H, W) * (H, 1) * (1, W)
        # This avoids allocating a full (H, W) weight matrix, saving memory
        weighted_image = image * profile_y[:, None] * profile_x[None, :]
        return weighted_image
    else:
        # In-place operation to avoid intermediate allocations
        if out.shape != image.shape:
            raise ValueError(f"Output shape {out.shape} does not match image shape {image.shape}")

        # Perform multiplication in steps
        # out = image * profile_y[:, None]
        np.multiply(image, profile_y[:, None], out=out)
        # out *= profile_x[None, :]
        np.multiply(out, profile_x[None, :], out=out)

        return out

def estimate_drift_2D(frame1, frame2, return_ccm = False):
    """
    Estimate the xy-drift between two 2D frames using cross-correlation.

    :param frame1: 2D numpy array, first frame
    :param frame2: 2D numpy array, second frame
    :return: Tuple (dx, dy), estimated drift in x and y directions
    """
    # Calculate the cross-correlation matrix
    # shift, error, diffphase = phase_cross_correlation(frame1, frame2)

    min_size_pixels = min(frame1.shape)

    frame1 = _2D_weighted_image(frame1, min_size_pixels // 3)
    frame2 = _2D_weighted_image(frame2, min_size_pixels // 3)


    frame1_max_proj_x = np.max(frame1, axis = 0)
    frame2_max_proj_x = np.max(frame2, axis = 0)

    frame1_max_proj_y = np.max(frame1, axis = 1)
    frame2_max_proj_y = np.max(frame2, axis = 1)

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
    # This speeds up the process by O(T) since each frame is accessed multiple times
    # and _2D_weighted_image allocates a full-size float32 array each time.
    min_size_pixels = min(x_shape, y_shape)
    overlap = min_size_pixels // 3

    # Pre-calculate weight profiles once
    profiles = _get_weight_profiles((x_shape, y_shape), overlap)

    # Store projections: (T, W) and (T, H)
    projections_x = []
    projections_y = []

    # Pre-allocate buffer for weighted frame to avoid repeated allocations
    # Determine dtype based on input (if int, result is float32 due to weights being float32)
    w_dtype = np.result_type(video_data.dtype, np.float32)
    w_frame_buffer = np.empty((x_shape, y_shape), dtype=w_dtype)

    # Allocate output buffer for subpixel shift to avoid reallocation in loop
    shift_output_buffer = np.empty_like(w_frame_buffer)

    # Iterate once to compute all projections
    for t in range(t_shape):
        w_frame = _2D_weighted_image(video_data[t], overlap, profiles=profiles, out=w_frame_buffer)
        projections_x.append(np.max(w_frame, axis=0))
        projections_y.append(np.max(w_frame, axis=1))

    # Convert to arrays for easy indexing
    projections_x = np.array(projections_x)
    projections_y = np.array(projections_y)

    # Loop through each time point in the video data, starting from the second frame
    # Wrap the range function with tqdm for a progress bar

    if reverse_time == 'both':
        # Forward order for 'both' (Bidirectional Estimation, Reference=Frame 0)
        range_values = range(1, t_shape)

        # Ensure first frame is copied
        corrected_data[0] = video_data[0]

        for time_point in tqdm(range_values, desc='Applying Drift Correction'):
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
            # (dx_backward - dx_forward) / 2
            dx = (dx_backward - dx_forward) / 2
            dy = (dy_backward - dy_forward) / 2


            ##### if too large, then keep as zero ####
            if abs(dx) > x_shape//5:
                dx = 0.0
            if abs(dy) > y_shape//5:
                # print('Whaa')
                dy = 0.0

            # Update the cumulative drift
            cum_dx, cum_dy = cum_dx + dx, cum_dy + dy

            # Apply drift correction to the current frame
            # NOTE: We cast to integer for the shift operation, but keep cumulative drift as float
            # to prevent integrator windup/loss of precision for slow drifts.

            # OFFSET = 0 for Forward iteration
            OFFSET = 0

            if method == 'subpixel':
                # Subpixel correction using bicubic interpolation
                s_dy, s_dx = cum_dy, cum_dx

                # Bolt: Reuse pre-allocated buffers to avoid memory churn.
                # w_frame_buffer is float32 (or similar) from pre-projection step.
                w_frame_buffer[:] = video_data[time_point - OFFSET]

                # Perform shift directly into pre-allocated output buffer
                shift(
                    w_frame_buffer,
                    shift=(s_dy, s_dx),
                    order=3,
                    mode='constant',
                    cval=min_value,
                    output=shift_output_buffer
                )

                # Robust clipping in-place
                if dtype_min is not None and dtype_max is not None:
                    np.clip(shift_output_buffer, dtype_min, dtype_max, out=shift_output_buffer)

                # Assign (implicit cast back to original dtype)
                corrected_data[time_point] = shift_output_buffer

            else:
                # Integer correction
                shift_dx = int(round(cum_dx))
                shift_dy = int(round(cum_dy))

                zero_shift_multi_dimensional(
                    video_data[time_point - OFFSET],
                    shifts=(shift_dy, shift_dx),
                    fill_value=min_value,
                    out=corrected_data[time_point]
                )

            # Record the drift values and cumulative drift for the current time point
            drift_records.append({'Time Point': time_point, 'dx': dx, 'dy': dy, 'cum_dx': cum_dx, 'cum_dy': cum_dy})
    else:

        if not reverse_time:
            # Regular order
            range_values = range(1, t_shape)
            DRIFT_SIGN = 1

            # The first frame does not need correction
            corrected_data[0] = video_data[0]
        else:
            # Reversed order
            range_values = range(t_shape - 1, 0, -1)
            DRIFT_SIGN = -1

            # The first frame does not need correction
            corrected_data[-1] = video_data[-1]

        for time_point in tqdm(range_values, desc='Applying Drift Correction'):
            # Estimate the drift between the current frame and the previous frame

            # Use precomputed projections with subpixel precision
            shift_x, _, _ = phase_cross_correlation(projections_x[time_point - 1], projections_x[time_point], upsample_factor=100)
            shift_y, _, _ = phase_cross_correlation(projections_y[time_point - 1], projections_y[time_point], upsample_factor=100)
            dx, dy = shift_x[0], shift_y[0]

            dx, dy = dx * DRIFT_SIGN, dy * DRIFT_SIGN

            ##### if too large, then keep as zero ####
            if abs(dx) > x_shape//5:
                dx = 0.0
            if abs(dy) > y_shape//5:
                # print('Whaa')
                dy = 0.0

            # Update the cumulative drift
            cum_dx, cum_dy = cum_dx + dx, cum_dy + dy

            # Apply drift correction to the current frame
            # NOTE: We cast to integer for the shift operation, but keep cumulative drift as float
            # to prevent integrator windup/loss of precision for slow drifts.

            OFFSET = 1 if reverse_time else 0 

            if method == 'subpixel':
                # Subpixel correction using bicubic interpolation
                s_dy, s_dx = cum_dy, cum_dx

                # Bolt: Reuse pre-allocated buffers to avoid memory churn.
                # w_frame_buffer is float32 (or similar) from pre-projection step.
                w_frame_buffer[:] = video_data[time_point - OFFSET]

                # Perform shift directly into pre-allocated output buffer
                shift(
                    w_frame_buffer,
                    shift=(s_dy, s_dx),
                    order=3,
                    mode='constant',
                    cval=min_value,
                    output=shift_output_buffer
                )

                if dtype_min is not None and dtype_max is not None:
                    np.clip(shift_output_buffer, dtype_min, dtype_max, out=shift_output_buffer)

                corrected_data[time_point] = shift_output_buffer

            else:
                shift_dx = int(round(cum_dx))
                shift_dy = int(round(cum_dy))

                zero_shift_multi_dimensional(
                    video_data[time_point - OFFSET],
                    shifts=(shift_dy, shift_dx),
                    fill_value=min_value,
                    out=corrected_data[time_point]
                )

            # Record the drift values and cumulative drift for the current time point
            drift_records.append({'Time Point': time_point, 'dx': dx, 'dy': dy, 'cum_dx': cum_dx, 'cum_dy': cum_dy})

    # Create a DataFrame from the list of drift records
    drift_table = pd.DataFrame(drift_records)
    drift_table.sort_values(by=['Time Point'], inplace=True)

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
