# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
# def color_coded_projection(image: np.ndarray, color_map = 'plasma') -> np.ndarray:

from matplotlib import colormaps as mpl_colormaps
import numpy as np
import matplotlib.pyplot as plt



def color_coded_projection(image: np.ndarray, color_map='plasma') -> np.ndarray:
    #     """
    #     Pseudocolors each frame of a 3D image (time, y, x) using the specified color map.

    #     :param image: 3D array (time, y, x) representing the image
    #     :param color_map: Name of the color map to use (default is 'plasma')
    #     :return: RGB image with dimensions (y, x, channel)
    #     """
    # Ensure the input image has three dimensions (time, y, x)
    if image.ndim != 3:
        raise ValueError("Input image must have three dimensions (time, y, x)")

    # Get the colormap
    cmap = mpl_colormaps[color_map]

    # Get the time dimension
    time_dimension = image.shape[0]

    # Initialize an empty array to store the colored frames (time, y, x, channels)
    colored_frames = np.zeros((time_dimension, image.shape[1], image.shape[2], 3), dtype=np.float32)

    # Loop through the time dimension and apply the colormap
    for t in range(time_dimension):
        # Normalize the current frame to the range [0, 1]
        frame = image[t]
        frame_min = frame.min()
        frame_max = frame.max()
        frame_normalized = (frame - frame_min) / (frame_max - frame_min) if frame_max > frame_min else frame

        # Get the corresponding color value from the colormap based on the current time index
        color_value = cmap(t / (time_dimension - 1))[:3]

        # Modulate the color value by the intensity factor of each pixel and store in the colored frames array
        for c in range(3): # Loop through the RGB channels
            colored_frames[t, :, :, c] = frame_normalized * color_value[c]

    # Take the maximum intensity projection across the time axis
    rgb_image = colored_frames.max(axis=0)

    return rgb_image
