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

def compute_drift_trajectory(projections_x, projections_y, mode='forward'):
    """
    Compute the global trajectory (position) of each frame relative to Frame 0
    by integrating pairwise drift estimates.

    :param projections_x: (T, W) array of x-projections (float32, windowed)
    :param projections_y: (T, H) array of y-projections (float32, windowed)
    :param mode: 'forward', 'reverse', or 'both'.
                 'reverse' is treated same as 'forward' for trajectory calculation,
                 but implies reference frame selection later.
                 'both' uses bidirectional estimation averaging.
    :return:
        - positions: (T, 2) array of (y, x) positions relative to Frame 0.
        - pairwise_steps: List of (dx, dy) steps for drift table reporting.
    """
    T = projections_x.shape[0]
    # projections_x is (T, W) -> W is width (x dim)
    # projections_y is (T, H) -> H is height (y dim)
    W = projections_x.shape[1]
    H = projections_y.shape[1]

    positions = np.zeros((T, 2)) # (y, x)
    pairwise_steps = []

    # Initialize frame 0 step
    pairwise_steps.append({'Time Point': 0, 'dx': 0.0, 'dy': 0.0})

    for t in tqdm(range(1, T), desc='Estimating Drift Trajectory'):
        # Forward Estimate: Shift to move T to T-1
        # pcc(ref, moving) returns shift

        shift_x_fwd, _, _ = phase_cross_correlation(
            projections_x[t-1], projections_x[t], upsample_factor=100
        )
        shift_y_fwd, _, _ = phase_cross_correlation(
            projections_y[t-1], projections_y[t], upsample_factor=100
        )
        dx, dy = shift_x_fwd[0], shift_y_fwd[0]

        if mode == 'both':
            # Backward Estimate: Shift to move T-1 to T
            shift_x_bwd, _, _ = phase_cross_correlation(
                projections_x[t], projections_x[t-1], upsample_factor=100
            )
            shift_y_bwd, _, _ = phase_cross_correlation(
                projections_y[t], projections_y[t-1], upsample_factor=100
            )
            dx_b, dy_b = shift_x_bwd[0], shift_y_bwd[0]

            # Average the forward step and negative backward step
            dx = (dx - dx_b) / 2
            dy = (dy - dy_b) / 2

        # Large shift filtering
        if abs(dx) > W / 5:
            dx = 0.0
        if abs(dy) > H / 5:
            dy = 0.0

        # Update Position
        # If dx is shift to move T -> T-1, then T is at Pos[T-1] - dx
        positions[t] = positions[t-1] - [dy, dx]

        pairwise_steps.append({
            'Time Point': t,
            'dx': dx,
            'dy': dy
        })

    return positions, pairwise_steps

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

    min_value = video_data.min()

    # Pre-calculate data range limits for clipping (to prevent interpolation ringing)
    if np.issubdtype(video_data.dtype, np.integer):
        dtype_min = np.iinfo(video_data.dtype).min
        dtype_max = np.iinfo(video_data.dtype).max
    else:
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
    projections_x = np.array(projections_x, dtype=np.float32)
    projections_y = np.array(projections_y, dtype=np.float32)

    # Apply 1D Tukey window to the entire stack
    win_x = windows.tukey(y_shape, alpha=0.1).astype(np.float32)
    win_y = windows.tukey(x_shape, alpha=0.1).astype(np.float32)

    # Broadcasting (T, N) * (1, N) -> (T, N)
    projections_x *= win_x[None, :]
    projections_y *= win_y[None, :]

    # Determine trajectory estimation mode
    traj_mode = 'both' if reverse_time == 'both' else 'forward'

    # Compute global trajectory
    # positions is (T, 2) array of (y, x)
    positions, pairwise_steps = compute_drift_trajectory(projections_x, projections_y, mode=traj_mode)

    # Determine reference frame position based on reverse_time
    if reverse_time is True or reverse_time == 'reverse':
        # Align everything to the last frame
        ref_pos = positions[-1]
    else:
        # Align everything to the first frame (0)
        # This covers 'forward' (False) and 'both'
        ref_pos = positions[0]

    # Calculate required shifts to align to ref_pos
    # Shift to apply = Ref - Pos
    # If Pos is (10, 10) and Ref is (0, 0), Shift is (-10, -10).
    shifts = ref_pos - positions # (T, 2) -> (dy, dx)

    drift_records = []

    # Apply corrections
    for t in tqdm(range(t_shape), desc='Applying Drift Correction'):
        s_dy, s_dx = shifts[t]

        # Populate drift record
        # Note: We use the pairwise steps for 'dx/dy' fields (step-wise drift)
        # and the calculated shift for 'cum_dx/cum_dy' (total correction)
        step = pairwise_steps[t]
        drift_records.append({
            'Time Point': t,
            'dx': step['dx'],
            'dy': step['dy'],
            'cum_dx': s_dx,
            'cum_dy': s_dy
        })

        if method == 'subpixel':
            input_frame = video_data[t].astype(np.float32)

            # Efficient subpixel shift
            shifted_slice = shift(
                input_frame,
                shift=(s_dy, s_dx),
                order=3,
                mode='constant',
                cval=min_value
            )

            if dtype_min is not None and dtype_max is not None:
                np.clip(shifted_slice, dtype_min, dtype_max, out=shifted_slice)

            corrected_data[t] = shifted_slice

        else:
            # Integer correction
            shift_dx = int(round(s_dx))
            shift_dy = int(round(s_dy))

            zero_shift_multi_dimensional(
                video_data[t],
                shifts=(shift_dy, shift_dx),
                fill_value=min_value,
                out=corrected_data[t]
            )

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
