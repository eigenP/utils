# /// script
# requires-python = ">=3.11"
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
import logging
import warnings

import numpy as np
import pandas as pd
from typing import Literal, Optional
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scipy.ndimage import shift as _ndi_shift
from scipy.signal import windows
# from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from ._conventions import warn_deprecated

logger = logging.getLogger(__name__)


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
    if isinstance(shifts, (int, np.integer)):
        shifts = [int(shifts)] * arr.ndim
    elif isinstance(shifts, (list, tuple, np.ndarray)):
        if len(shifts) != arr.ndim:
            raise ValueError("Length of shifts must be equal to the number of dimensions in the array.")
        if not all(isinstance(shift, (int, np.integer)) for shift in shifts):
            raise TypeError("All shift values must be integers.")
        shifts = [int(shift) for shift in shifts]
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

def estimate_drift(frame1, frame2, return_ccm=False, order=None):
    """
    Estimate the correction that registers `frame2` onto `frame1`.

    Each axis is estimated from the frames' max projections onto that axis (the
    baseline removed, a Tukey(alpha=0.1) window applied) with
    `estimate_shift_1d_iterative`, the estimator `apply_drift_correction` uses.
    The result is the shift to *apply* to `frame2` (the negative of its motion
    relative to `frame1`).

    :param frame1: reference frame, 2D (Y, X) (or 3D (Z, Y, X) with order='zyx')
    :param frame2: frame to register, same shape
    :param return_ccm: deprecated; also return the Y-profile cross-correlation map
    :param order: 'zyx' returns the correction in array-axis order ((dy, dx), or
        (dz, dy, dx) for 3D), which `apply_subpixel_drift_correction` takes directly;
        'xy' returns (dx, dy) (2D only). The default (None) is 'xy' with a
        FutureWarning: it becomes 'zyx' in eigenp_utils 0.1.0.
    :return: ndarray of the correction, in the order above

    Changed (D22, R7-03, R7-13, 2026-11): uses the iterative estimator (the single
    windowed correlation erred by up to ~0.9 px); new `order=` keyword (the old
    (x, y) order is the default for one release, with a FutureWarning; chaining it
    into `apply_subpixel_drift_correction` swapped the axes); `return_ccm` is
    deprecated (FutureWarning, removed in 0.1.0).
    """
    if order is None:
        warnings.warn(
            "estimate_drift returns (dx, dy) by default; the default will become "
            "order='zyx' (array-axis order, as apply_subpixel_drift_correction takes) "
            "in eigenp_utils 0.1.0. Pass order='zyx' (or order='xy' for the old order).",
            FutureWarning, stacklevel=2
        )
        order = 'xy'
    if order not in ('xy', 'zyx'):
        raise ValueError(f"order must be 'xy' or 'zyx', got {order!r}.")

    frame1 = np.asarray(frame1)
    frame2 = np.asarray(frame2)
    if frame1.shape != frame2.shape:
        raise ValueError(f"frame1 and frame2 must have the same shape, got {frame1.shape} and {frame2.shape}.")
    if order == 'xy' and frame1.ndim != 2:
        raise ValueError(f"order='xy' needs 2D frames, got {frame1.ndim}D; use order='zyx'.")
    if frame1.ndim not in (2, 3):
        raise ValueError(f"Expected 2D (Y, X) or 3D (Z, Y, X) frames, got {frame1.ndim}D.")

    # matth: Decoupled 1D Windowing
    # Instead of windowing the image (which couples the axes' attenuation and causes
    # information loss at the boundaries), the raw max projections onto each axis
    # are windowed in 1D; the window is stationary, so the iterative estimator
    # re-aligns the moving profile before each refinement.
    correction = []
    profiles = []
    for axis in range(frame1.ndim):
        other = tuple(a for a in range(frame1.ndim) if a != axis)
        p1 = np.max(frame1, axis=other)
        p2 = np.max(frame2, axis=other)
        win = windows.tukey(p1.shape[0], alpha=0.1).astype(np.float32)
        correction.append(float(estimate_shift_1d_iterative(p1, p2, win)))
        profiles.append((p1, p2, win))
    correction = np.array(correction)
    if order == 'xy':
        correction = correction[::-1]

    if return_ccm:
        warn_deprecated("estimate_drift(return_ccm=True)", stacklevel=3)
        # The cross-correlation of the (baseline-removed, windowed) Y profiles,
        # stacked twice: shape (2, H)
        p1, p2, win = profiles[-2]
        prof1 = _remove_baseline(p1) * win
        prof2 = _remove_baseline(p2) * win
        f1 = np.vstack((prof1, prof1))
        f2 = np.vstack((prof2, prof2))
        image_product = np.fft.fft2(f1) * np.fft.fft2(f2).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        return correction, cc_image.real
    return correction

def _remove_baseline(profile):
    """
    Subtract a projection's background level (its minimum).

    A camera offset or background is a constant pedestal in every max projection.
    Multiplied by the (stationary) window it becomes a large feature shared by both
    profiles at zero shift, which pulls the correlation peak towards zero (with phase
    correlation, offset 500 read a shift of +6.7 as 0). The minimum (not the median)
    leaves no negative pedestal on zero-background data. Integers become float32.
    """
    profile = np.asarray(profile)
    if not np.issubdtype(profile.dtype, np.floating):
        profile = profile.astype(np.float32)
    return profile - profile.min()


def estimate_shift_1d_iterative(ref_proj, moving_proj, window=None, max_iter=3, tol=0.01):
    """
    Estimates 1D shift using iterative windowed cross-correlation to eliminate stationary window bias.

    0. Remove each profile's baseline (minimum), so a background offset does not bias the shift.
    1. Estimate shift d.
    2. Shift moving signal by -d (aligning with ref).
    3. Re-estimate residual shift.
    4. Repeat until convergence.

    `window=None` uses no window (all ones).
    """
    ref_proj = _remove_baseline(ref_proj)
    moving_proj = _remove_baseline(moving_proj)
    if window is None:
        window = np.ones_like(ref_proj)

    ref_win = ref_proj * window
    if not np.any(ref_win) or not np.any(moving_proj):
        return 0.0

    current_shift = 0.0

    for _ in range(max_iter):
        shift_val = current_shift

        # Apply shift to align moving with ref
        # Using cubic interpolation (order=3) for accuracy; cval=0 is the baseline
        # level after _remove_baseline, so no artificial step is shifted in
        if abs(shift_val) > 0.001:
            aligned_mov = shift(moving_proj, shift_val, order=3, mode='constant', cval=0)
        else:
            aligned_mov = moving_proj

        # Estimate residual shift
        # We multiply by window here. The window is stationary (matched to ref).
        # Since aligned_mov is aligned to ref, the window is correctly placed for both.
        # Plain cross-correlation (normalization=None), not phase correlation: whitening
        # the spectrum gives the noise-dominated high frequencies of a max projection as
        # much weight as the signal, so Poisson noise produced spurious peaks once the
        # background pedestal (which had anchored the peak at zero) is removed.
        mov_win = aligned_mov * window
        if not np.any(mov_win):
            break
        res, _, _ = phase_cross_correlation(ref_win, mov_win, upsample_factor=100,
                                            normalization=None)
        res = res[0]

        current_shift += res

        # Convergence check
        if abs(res) < tol:
            break

    return current_shift

def _check_max_step(max_step):
    """Raise ValueError unless `max_step` is None or a finite number > 0."""
    if max_step is None:
        return
    if isinstance(max_step, (bool, np.bool_)) or not np.isscalar(max_step) \
            or not np.isreal(max_step) or not np.isfinite(max_step) or max_step <= 0:
        raise ValueError(
            f"max_step must be None or a finite number > 0 (a fraction of the axis "
            f"below 1, pixels from 1), got {max_step!r}."
        )


def compute_drift_trajectory(
    projections: list,
    mode: Literal['forward', 'reverse', 'both'] = 'forward',
    shape_limit: tuple = None,
    windows: tuple = None,
    max_step: Optional[float] = 0.2,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Computes the drift trajectory (cumulative corrections) for the video.

    :param projections: List of numpy arrays, one for each spatial dimension (e.g., [proj_z, proj_y, proj_x]).
    :param mode: 'forward', 'reverse', or 'both'
    :param shape_limit: Tuple of the axis lengths (e.g., (z_shape, y_shape, x_shape)) that
        `max_step` fractions refer to; None disables fractional limits.
    :param windows: Tuple of 1D windows to apply to each projection (e.g., (win_z, win_y, win_x))
    :param max_step: Largest plausible step between consecutive frames. A value below 1
        is a fraction of the axis length (default 0.2: max(axis // 5, 1) px); 1 or more is
        in pixels; None disables the check. A larger step is treated as an outlier and
        set to 0, with a UserWarning naming the time point, axis and value. The
        trajectory is cumulative, so every later frame then stays offset by that step.
        Anything else (0, negative, NaN, inf, a string) raises ValueError.
    :param progress: show a tqdm progress bar.
    Returns a DataFrame with columns: ['Time Point', 'dx', 'dy', 'dz', 'cum_dx', 'cum_dy', 'cum_dz']
    where cum_d* are the shifts needed to align the frame to the reference.
    If 2 spatial dimensions are provided, dz and cum_dz will be 0.

    Changed (D22, R7-04, R7-09, 2026-11): a rejected step warns (it used to be set
    to 0 silently); new `max_step` (the old fixed rule is the default 0.2) and
    `progress` parameters.
    """
    num_dims = len(projections)
    T = projections[0].shape[0]
    axis_names = ('Z', 'Y', 'X')[-num_dims:]

    # Initialize steps and cumulative drift for 3 axes (z, y, x)
    # If 2D (num_dims == 2), we will just map index 0 -> y, index 1 -> x
    # If 3D (num_dims == 3), index 0 -> z, index 1 -> y, index 2 -> x
    cum_d = np.zeros((3, T), dtype=np.float32)
    d = np.zeros((3, T), dtype=np.float32)

    wins = windows if windows else [None] * num_dims

    # Largest accepted step per axis (None: no limit). A fraction of the axis is
    # floored to whole pixels and is at least 1 px, so short axes (e.g. Z < 5) are
    # still corrected; without shape_limit a fraction has nothing to refer to.
    _check_max_step(max_step)
    if max_step is None:
        step_limits = [None] * num_dims
    elif max_step < 1:
        if shape_limit:
            step_limits = [max(int(np.floor(n * max_step + 1e-9)), 1) for n in shape_limit]
        else:
            step_limits = [None] * num_dims
    else:
        step_limits = [float(max_step)] * num_dims

    # Helper to reject implausible drift steps as outliers
    def clamp(val, d_idx, t):
        limit = step_limits[d_idx]
        if limit is not None and abs(val) > limit:
            warnings.warn(
                f"Drift step at time point {t}, axis {axis_names[d_idx]}: {val:.2f} px "
                f"exceeds max_step ({limit:g} px) and was rejected (set to 0); every later "
                f"frame keeps this offset. Raise max_step or pass max_step=None to keep it.",
                UserWarning, stacklevel=3
            )
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
        for t in tqdm(range(T - 1, 0, -1), desc='Computing Drift Trajectory (Reverse)',
                      disable=not progress):
            shifts = align_step(t, t-1)
            for d_idx in range(num_dims):
                c = clamp(shifts[d_idx], d_idx, t - 1)
                # target axis in output arrays:
                # If 2D (y, x), d_idx=0 -> y (axis 1), d_idx=1 -> x (axis 2)
                # If 3D (z, y, x), d_idx=0 -> z (axis 0), d_idx=1 -> y (axis 1), d_idx=2 -> x (axis 2)
                out_idx = d_idx + (3 - num_dims)
                d[out_idx, t-1] = c
                cum_d[out_idx, t-1] = cum_d[out_idx, t] + c

    else:
        # Forward or Both
        for t in tqdm(range(1, T), desc=f'Computing Drift Trajectory ({mode})',
                      disable=not progress):
            if mode == 'both':
                shifts_bwd = align_step(t-1, t)
                shifts_fwd = align_step(t, t-1)
                shifts = [(b - f) / 2 for b, f in zip(shifts_bwd, shifts_fwd)]
            else:
                shifts = align_step(t-1, t)

            for d_idx in range(num_dims):
                c = clamp(shifts[d_idx], d_idx, t)
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
    method: Literal['integer', 'subpixel'] = 'integer',
    max_step: Optional[float] = 0.2,
    progress: bool = True,
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
    :param max_step: largest plausible step between frames, see `compute_drift_trajectory`
        (default 0.2 of each axis; None disables the outlier rejection).
    :param progress: show tqdm progress bars.
    :return: A tuple containing two elements:
        - corrected_data: A numpy array of the same shape as video_data.
        - drift_table: A pandas DataFrame containing the drift values.

    Changed (D22, R7-04, R7-09, R7-10, 2026-11): messages go to the module logger
    (``logging.getLogger(__name__)``; the root logger is no longer configured);
    new `max_step` and `progress` parameters; a rejected step warns; an unknown
    `method` raises ValueError (it silently used 'integer').
    """
    if method not in ('integer', 'subpixel'):
        raise ValueError(f"method must be 'integer' or 'subpixel', got {method!r}.")
    _check_max_step(max_step)
    ndim = video_data.ndim
    if ndim == 3:
        logger.info("Input is 3D (T, Y, X). Proceeding with 2D+t max-projection registration.")
        t_shape, y_shape, x_shape = video_data.shape
        num_spatial_dims = 2
    elif ndim == 4:
        logger.info("Input is 4D (T, Z, Y, X). Proceeding with 3D+t max-projection registration.")
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
        windows=windows_tuple,
        max_step=max_step,
        progress=progress,
    )

    # Iterate all frames to apply correction
    for t in tqdm(range(t_shape), desc='Applying Drift Correction', disable=not progress):
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
                # Round, don't truncate, when storing integers (truncation biased values ~0.5 DN low)
                np.rint(shifted_slice, out=shifted_slice)
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

def apply_subpixel_drift_correction(image, shift=None, *, drift=None):
    """
    Applies subpixel drift correction to an image using bicubic interpolation.

    Parameters:
    image (np.array): The input image.
    shift (tuple): The shift (correction) to apply, in pixels, in array-axis order
        ((dy, dx) for 2D, (dz, dy, dx) for 3D): e.g.
        ``estimate_drift(frame1, frame2, order='zyx')``.
    drift: deprecated keyword alias of `shift`.

    Returns:
    np.array: The corrected image.

    Changed (D22, R7-03, 2026-11): the parameter is named `shift` (it is the
    correction to apply, not the measured drift); ``drift=`` still works and
    emits a FutureWarning. Positional calls are unchanged.
    """
    if drift is not None:
        if shift is not None:
            raise TypeError("Pass either shift or the deprecated drift, not both.")
        warn_deprecated("apply_subpixel_drift_correction(drift=...)", "shift=", stacklevel=3)
        shift = drift
    if shift is None:
        raise TypeError("apply_subpixel_drift_correction() missing required argument: 'shift'")
    # Ensure image is a numpy array
    image = np.asarray(image)
    min_value = image.min()

    # Efficiently apply subpixel shift using scipy.ndimage.shift
    # This avoids creating a full coordinate grid (O(N) memory savings)
    corrected_image = _ndi_shift(image, shift=shift, order=3, mode='constant', cval=min_value)

    return corrected_image
