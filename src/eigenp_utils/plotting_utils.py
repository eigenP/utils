# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib import colormaps as mpl_colormaps
from pathlib import Path

# --- Initialization: Load Font and Style ---
ROOT_DIR = Path(__file__).parent
font_path = ROOT_DIR / 'Inter-Regular.ttf'
style_path = ROOT_DIR / 'scientific.mplstyle'

# Load Style
try:
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        print(f"Warning: Style file not found at {style_path}")
except Exception as e:
    print(f"Warning: Failed to load style from {style_path}: {e}")

# Load Font
try:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        prop = font_manager.FontProperties(fname=str(font_path))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = prop.get_name()
    else:
        # Fallback if file not found, though it should be there
        print(f"Warning: Font file not found at {font_path}")
except Exception as e:
    print(f"Warning: Failed to load font from {font_path}: {e}")


# --- Initialization: Create labels_cmap ---
# Get the original tab20 colors
tab20 = plt.get_cmap('tab20')
original_colors = tab20.colors  # This retrieves the distinct colors used in tab20

# Interpolate to create more colors
num_new_colors = 254  # One less because we'll add black
new_colors = np.linspace(0, 1, num_new_colors)

# Using a linear interpolation to blend between existing colors
# Note: original code used list comprehension with tab20(x), but tab20 is discrete.
# LinearSegmentedColormap.from_list handles interpolation if we just give it colors.
# But the user provided specific logic:
interpolated_colors = [tab20(x) for x in new_colors]

# Randomly mix the colors and add black as the first color
# Set the seed locally to avoid affecting global numpy state
rng = np.random.RandomState(42)

rng.shuffle(interpolated_colors)
final_colors = [(0, 0, 0)] + interpolated_colors  # Black at the zero index

# Create the new colormap
labels_cmap = LinearSegmentedColormap.from_list("labels_cmap", final_colors)


def hist_imshow(image, bins=64, gamma = 1, return_image_only = False,  **imshow_kwargs):
    """
    Displays an image and its histogram.

    This function processes a given image stack, ensuring it is in a 2D format suitable for display. If the
    image stack has more than two dimensions, the function extracts the middle slice from each of the extra
    dimensions and then displays this 2D slice. Alongside the image, the function plots a histogram of the
    pixel intensities to provide a visual representation of the distribution of pixel values within the image.

    Parameters
    ----------
    image : array-like
        The input image. Can be multidimensional, but only a 2D slice (from the middle of any extra
        dimensions) will be displayed.
    bins : int, optional
        The number of bins to use for the histogram. Default is 256.

    Returns
    -------
    dict
        A dictionary with two entries:
        ``"fig"`` containing the created :class:`matplotlib.figure.Figure` and
        ``"axes"`` containing a mapping of axis names to :class:`matplotlib.axes.Axes`
        objects. When ``return_image_only`` is ``True`` this function returns only
        the 2D image slice.

    Notes
    -----
    Additional information about the original shape and data type of the image is displayed on the x-axis label
    of the histogram.

    Examples
    --------
    >>> img = np.random.rand(100, 100)  # Generate a random image
    >>> res = hist_imshow(img)
    >>> res["fig"].show()  # Display the figure with the image and its histogram
    """

    ### SET DEFAULT KWARGS
    # Ensure 'origin' is set to 'lower' if not specified
    imshow_kwargs.setdefault('origin', 'lower')
    imshow_kwargs.setdefault('cmap', 'gray')

    # Ensure image is 2D so that we can plot it
    im_shape = image.shape
    print(f'Image shape: {im_shape}')
    if len(im_shape) > 2:
        # Calculate the middle index for each dimension except the last two
        middle_indices = [s // 2 for s in im_shape[:-2]]

        # Add slice(None) for the last two dimensions
        indexing = middle_indices + [slice(None), slice(None)]
        slices = tuple(indexing)

        print('Displaying only the last two dims (of the "middle" slices)')
        # print(slices)
        image = image[slices]

    if return_image_only:
        return image

    fig, axes = plt.subplot_mosaic([['Image', '.'], ['Image', 'Histogram'], ['Image', '.']],
                                   layout = 'constrained')

    norm = None
    if gamma != 1:
        norm = PowerNorm(gamma=gamma)


    axes['Image'].imshow(image, norm = norm, interpolation = 'nearest', **imshow_kwargs)

    # Display histogram
    axes['Histogram'].hist((image.ravel()), bins=bins, density = True, histtype='stepfilled')
    axes['Histogram'].set_yscale('log')
    # axes['Histogram'].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    min_val = image.min()
    max_val = image.max()
    mean_val = image.mean()
    stats_msg = f"min:{min_val:.3g} max:{max_val:.3g} mean:{mean_val:.3g}"
    axes['Histogram'].set_xlabel(
        f' Pixel intensity ({stats_msg}) \n \n shape: {im_shape} \n dtype: {image.dtype}'
    )
    axes['Histogram'].set_ylabel('Log Frequency')

    # axes['Histogram'].set_xlim(0, 1)
    # axes['Histogram'].set_yticks([])


    return {"fig": fig, "axes": axes}


def color_coded_projection(image: np.ndarray, color_map='plasma') -> np.ndarray:
    """
    Pseudocolors each frame of a 3D image (time, y, x) using the specified color map.

    :param image: 3D array (time, y, x) representing the image
    :param color_map: Name of the color map to use (default is 'plasma')
    :return: RGB image with dimensions (y, x, channel)
    """
    # Ensure the input image has three dimensions (time, y, x)
    if image.ndim != 3:
        raise ValueError("Input image must have three dimensions (time, y, x)")

    # Get the colormap
    cmap = mpl_colormaps[color_map]

    # Get the time dimension
    time_dimension = image.shape[0]

    # Initialize the result array (y, x, 3) for the maximum intensity projection
    rgb_image = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.float32)

    # Loop through the time dimension and apply the colormap
    for t in range(time_dimension):
        # Normalize the current frame to the range [0, 1]
        frame = image[t]
        frame_min = frame.min()
        frame_max = frame.max()
        frame_normalized = (frame - frame_min) / (frame_max - frame_min) if frame_max > frame_min else frame

        # Get the corresponding color value from the colormap based on the current time index
        color_value = cmap(t / (time_dimension - 1))[:3]

        # Modulate the color value by the intensity factor and update max projection
        # Optimization: We update rgb_image iteratively to avoid allocating the large (time, y, x, 3) array.
        # We also loop over channels to avoid allocating a full (y, x, 3) temporary for the current frame.
        for c in range(3):  # Loop through the RGB channels
            # Compute this channel's contribution for the current time point
            # This allocates a temporary (y, x) array which is much smaller
            channel_component = frame_normalized * color_value[c]

            # Update the max projection in-place
            np.maximum(rgb_image[:, :, c], channel_component, out=rgb_image[:, :, c])

    return rgb_image





# Function to adjust colormaps
def adjust_colormap(cmap_name, start_ = 0.25, end_ = 1.0, start_color = [0.9, 0.9, 0.9, 1]):
    '''
        # Use colormap from 25% to 100% (just the color part)
        # start_color = [0.9, 0.9, 0.9, 1] # gray
    '''
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(start_, end_, 256))  # Use colormap from 25% to 100% (just the color part)
    colors = np.vstack((start_color, colors))  # Adding faint gray for underflow
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_adjusted", colors)
    return new_cmap
