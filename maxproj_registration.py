#@markdown `maxproj_registration.py`

### Imports
import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
# from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm



def zero_shift_multi_dimensional(arr, shifts = 0, fill_value=0):
    """
    Shift the elements of a multi-dimensional NumPy array along each axis by specified amounts, filling the vacant positions with a specified fill value.

    :param arr: A multi-dimensional NumPy array
    :param shifts: A single integer or a list/tuple of integers specifying the shift amounts for each axis
    :param fill_value: An optional value to fill the vacant positions after the shift (default is 0)
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

    # Initialize the result array with the fill value
    result = np.full_like(arr, fill_value)
    # Initialize slices for input and output arrays
    slices_input = [slice(None)] * arr.ndim
    slices_output = [slice(None)] * arr.ndim

    # Apply the shifts
    for axis, shift in enumerate(shifts):
        if shift > 0:
            slices_input[axis] = slice(None, -shift)
            slices_output[axis] = slice(shift, None)
        elif shift < 0:
            slices_input[axis] = slice(-shift, None)
            slices_output[axis] = slice(None, shift)

    # Perform the shift and fill in the result array
    result[tuple(slices_output)] = arr[tuple(slices_input)]
    return result


def estimate_drift_2D(frame1, frame2, return_ccm = False):
    """
    Estimate the xy-drift between two 2D frames using cross-correlation.

    :param frame1: 2D numpy array, first frame
    :param frame2: 2D numpy array, second frame
    :return: Tuple (dx, dy), estimated drift in x and y directions
    """
    # Calculate the cross-correlation matrix
    # shift, error, diffphase = phase_cross_correlation(frame1, frame2)
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
                                                      frame2_max_proj_x)

    shift_y, error, diffphase = phase_cross_correlation(frame1_max_proj_y,
                                                      frame2_max_proj_y)

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


def apply_drift_correction_2D(video_data, save_drift_table=False, csv_filename='drift_table.csv'):
    """
    Apply drift correction to video data.

    This function corrects for drift in video data frame by frame. It calculates the drift between
    consecutive frames using the `estimate_drift` function, and applies corrections to align the frames.
    The cumulative drift is also calculated and stored. Optionally, a table of drift values can be saved
    to a CSV file.

    :param video_data: A 3D numpy array representing the video data. The dimensions should be (time, x, y).
    :param save_drift_table: A boolean indicating whether to save the drift values to a CSV file. Default is False.
    :param csv_filename: The name of the CSV file to save the drift table to. Default is 'drift_table.csv'.
    :return: A tuple containing two elements:
        - corrected_data: A 3D numpy array of the same shape as video_data, representing the drift-corrected video.
        - drift_table: A pandas DataFrame containing the drift values, cumulative drift, and time points.
    """


    # Get the dimensions of the video data
    t, x, y = video_data.shape

    # Initialize an array to store the corrected video data
    corrected_data = np.zeros_like(video_data)

    # The first frame does not need correction
    corrected_data[0] = video_data[0]

    # Initialize variables to store cumulative drift
    cum_dx, cum_dy = 0, 0

    # Initialize a list to store drift records for each time point
    drift_records = []


    min_value = video_data.min()

    # Loop through each time point in the video data, starting from the second frame
    # Wrap the range function with tqdm for a progress bar
    for time_point in tqdm(range(1, t), desc='Applying Drift Correction'):
    # for time_point in range(1, t):
        # Estimate the drift between the current frame and the previous frame
        dx, dy = estimate_drift_2D(video_data[time_point - 1], video_data[time_point])

        ##### if too large, then keep as zero ####
        if abs(dx) > x//4:
            dx = 0
        if abs(dy) > y//4:
            print('Whaa')
            dy = 0

        # Update the cumulative drift
        cum_dx, cum_dy = int(cum_dx + dx), int(cum_dy + dy)

        # Apply drift correction to the current frame
        corrected_data[time_point] = zero_shift_multi_dimensional(video_data[time_point], shifts=(cum_dy, cum_dx), fill_value = min_value)

        # Record the drift values and cumulative drift for the current time point
        drift_records.append({'Time Point': time_point, 'dx': dx, 'dy': dy, 'cum_dx': cum_dx, 'cum_dy': cum_dy})

    # Create a DataFrame from the list of drift records
    drift_table = pd.DataFrame(drift_records)

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

    # Generate the coordinates grid
    coords = np.array(np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing='ij'
    ), dtype=float)  # change to float type


    # Apply the drift
    for i in range(3):  # loop over x, y, z
        coords[i, ...] -= drift[i]

    # Perform bicubic interpolation
    corrected_image = map_coordinates(image, coords, order=3, mode='constant', cval = min_value )

    return corrected_image
