# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pandas",
# ]
# ///
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib import colormaps as mpl_colormaps
from pathlib import Path
import pandas as pd
import itertools

# --- Initialization: Load Font and Style ---
ROOT_DIR = Path(__file__).parent
font_path = ROOT_DIR / 'Inter-Regular.ttf'
style_path = ROOT_DIR / 'scientific.mplstyle'

# Load Style

def set_plotting_style():
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


def raincloud_plot(data,
                   x_label=None,
                   y_label=None,
                   title=None,
                   ax=None,
                   orientation='vertical',
                   palette=None,
                   figsize=(4, 4)):
    """
    Creates a raincloud plot (half-violin + boxplot + jittered scatter).

    Parameters
    ----------
    data : array-like, dict, or pd.DataFrame
        Input data. Can be:
        - Single array/list.
        - List of arrays/lists.
        - Dictionary {label: array}.
        - DataFrame (columns are groups).
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Orientation of the plot.
    palette : list of colors or str, optional
        Colors to use.
    figsize : tuple, default (4, 4)
        Figure size if creating a new figure.

    Returns
    -------
    dict
        {'fig': figure, 'axes': ax}
    """

    # --- 1. Normalize Input Data ---
    plot_data = [] # List of (label, values)

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            vals = data[col].dropna().values
            plot_data.append((str(col), vals))
    elif isinstance(data, dict):
        for label, values in data.items():
            vals = np.asarray(values)
            vals = vals[~np.isnan(vals)] # Remove NaNs
            plot_data.append((str(label), vals))
    elif isinstance(data, (list, np.ndarray)):
        # Check if it's a list of arrays (multiple groups) or a single array
        data_arr = np.asarray(data, dtype=object)

        # Logic to determine if it's a single group or multiple:
        # 1. If it's a 1D array of numbers -> Single group
        # 2. If it's a list of lists/arrays -> Multiple groups
        # 3. If it's a 2D array -> Multiple groups (rows or cols? assume rows are groups)

        is_multiple = False

        # Try to convert to a numeric array
        try:
            numeric_arr = np.asarray(data, dtype=float)
            # If successful conversion
            if numeric_arr.ndim > 1:
                # 2D+ array -> treat as multiple groups (iterate over first dimension)
                is_multiple = True
            else:
                # 1D array -> single group
                is_multiple = False
        except (ValueError, TypeError):
            # Could not convert to a regular float array (likely jagged list of lists)
            # Treat as multiple groups
            is_multiple = True

        if is_multiple:
            for i, d in enumerate(data):
                vals = np.asarray(d, dtype=float)
                vals = vals[~np.isnan(vals)]
                plot_data.append((str(i), vals))
        else:
            # Single group
            vals = np.asarray(data, dtype=float)
            vals = vals[~np.isnan(vals)]
            plot_data.append(("Data", vals))
    else:
        raise ValueError("Unsupported data type")

    # --- 2. Setup Figure/Axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- 3. Colors ---
    n_groups = len(plot_data)
    if palette is None:
        # Default color
        colors = ["#4C78A8"] * n_groups
    elif isinstance(palette, str):
         # If single color string
        colors = [palette] * n_groups
    else:
        # List of colors
        colors = list(itertools.islice(itertools.cycle(palette), n_groups))

    # --- 4. Plotting Loop ---
    vert = (orientation == 'vertical')

    # Iterate over groups
    for i, (label, vals) in enumerate(plot_data):
        col = colors[i]
        pos = i # Position on categorical axis

        # A. Violin
        parts = ax.violinplot(vals, positions=[pos], widths=0.8,
                              showextrema=False, vert=vert)

        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(0.8)

            # Clipping to half
            # get_paths()[0].vertices is (N, 2) array of (x, y)
            verts = pc.get_paths()[0].vertices

            if vert:
                # Vertical: x is dim 0. Center is 'pos'.
                # To keep left half, we want x <= mean_x (or pos).
                # The violinplot draws symmetrically around 'pos'.
                # mean_x should be approx 'pos'.
                # We clip the Right side to the center, effectively removing it?
                # Original code: verts[:, 0] = np.clip(verts[:, 0], None, mean_x)
                # This clips x to (-inf, mean). So it keeps the left side.
                mean_val = verts[:, 0].mean()
                verts[:, 0] = np.clip(verts[:, 0], None, mean_val)
            else:
                # Horizontal: y is dim 1. Center is 'pos'.
                # Violin is drawn along y-axis centered at pos.
                # User wants violin "atop" (Y >= pos).
                # To keep "top" half (y >= mean), we clip (mean, None).
                mean_val = verts[:, 1].mean()
                verts[:, 1] = np.clip(verts[:, 1], mean_val, None)

        # B. Scatter (Rain)
        rng = np.random.default_rng(0)
        jitter_vals = rng.normal(loc=0, scale=0.04, size=len(vals))

        # Scatter needs to be offset to the "right" (or top) of the violin
        # Violin is on the "left" (or bottom)
        offset = 0.20

        if vert:
            x_scatter = pos + offset + jitter_vals
            y_scatter = vals
        else:
            x_scatter = vals
            # Horizontal: Violin is "atop" (above), so Scatter is "below" (pos - offset)
            y_scatter = pos - offset + jitter_vals

        ax.scatter(x_scatter, y_scatter, color=col, alpha=0.50,
                   edgecolor="black", linewidth=0.5, s=40)

        # C. Boxplot Elements (Median + IQR)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        if vert:
            # Vertical line for IQR at x=pos
            ax.vlines(pos, q1, q3, color="k", linewidth=5, zorder=2)
            # Dot for median
            ax.scatter(pos, med, color="white", edgecolor="k", linewidth=2, s=90, zorder=3)
        else:
            # Horizontal line for IQR at y=pos
            ax.hlines(pos, q1, q3, color="k", linewidth=5, zorder=2)
            ax.scatter(med, pos, color="white", edgecolor="k", linewidth=2, s=90, zorder=3)

    # --- 5. Cosmetics ---
    labels = [p[0] for p in plot_data]
    ticks = np.arange(len(plot_data))

    if vert:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        if x_label: ax.set_xlabel(x_label) # Use xlabel if provided (original used xticklabels header)
        if y_label: ax.set_ylabel(y_label)
    else:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return {"fig": fig, "axes": ax}


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
