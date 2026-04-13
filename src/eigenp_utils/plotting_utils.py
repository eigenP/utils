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
import math
import datetime

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
from .labels_cmap_data import LABELS_CMAP_COLORS

# Generate 255 alpha values linearly spaced from 0.8 to 1.0
alphas = np.linspace(0.8, 1.0, 255)

# Start with a transparent black at index 0
final_colors = [(0, 0, 0, 0)]

# Add the 255 pre-calculated Glasbey-style maximally distant colors
for i, (r, g, b) in enumerate(LABELS_CMAP_COLORS):
    # i goes from 0 to 254
    final_colors.append((r, g, b, alphas[i]))

# Create the new colormap
labels_cmap = LinearSegmentedColormap.from_list("labels_cmap", final_colors, N=256)

try:
    mpl_colormaps.register(labels_cmap, name="labels_cmap", force=True)
except AttributeError:
    # Fallback for older Matplotlib versions
    plt.register_cmap(name="labels_cmap", cmap=labels_cmap)

# Print a hint for users who might need the background to be black
print("Hint: labels_cmap background is transparent by default. To set it to black, run:")
print("labels_cmap._init(); labels_cmap._lut[0] = [0, 0, 0, 1]")


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
                   figsize=(4, 4),
                   x=None, y=None, hue=None, dodge=None,
                   size_scatter=40,
                   size_median=90,
                   alpha_scatter=0.50,
                   alpha_violin=0.8,
                   linewidth_scatter=0.5,
                   linewidth_boxplot=5,
                   offset_scatter=0.20):
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
    x : str, optional
        If data is a DataFrame, column name for x-axis variable.
    y : str, optional
        If data is a DataFrame, column name for y-axis variable.
    hue : str, optional
        If data is a DataFrame, column name for grouping variable.
    dodge : bool, optional
        When hue is nested, whether to draw elements at different positions.
        If None, defaults to True unless hue matches x or y variable.

    Returns
    -------
    dict
        {'fig': figure, 'axes': ax}
    """

    # --- 1. Normalize Input Data ---
    # We will convert all inputs into a list of "PlotItems".
    # Each PlotItem is a dict: {'values': array, 'position': float, 'color': Any, 'width': float, 'label': str}

    plot_items = []
    xtick_labels = []
    xtick_positions = []

    vert = (orientation == 'vertical')

    # Branch A: DataFrame with x/y specified (Seaborn-style)
    if isinstance(data, pd.DataFrame) and (x is not None or y is not None):
        if vert:
            cat_col = x
            val_col = y
            if val_col is None: raise ValueError("For vertical plot, y must be specified (as value).")
        else:
            cat_col = y
            val_col = x
            if val_col is None: raise ValueError("For horizontal plot, x must be specified (as value).")

        # Check columns exist
        if val_col and val_col not in data.columns: raise ValueError(f"Column {val_col} not found.")
        if cat_col and cat_col not in data.columns: raise ValueError(f"Column {cat_col} not found.")
        if hue and hue not in data.columns: raise ValueError(f"Column {hue} not found.")

        # Get Data and Drop NaNs
        cols_to_check = [c for c in [val_col, cat_col, hue] if c]
        df_clean = data.dropna(subset=cols_to_check)

        # Get Categories
        if cat_col:
            cats = df_clean[cat_col].unique()
            # Try to respect pandas categorical order if present
            if isinstance(df_clean[cat_col].dtype, pd.CategoricalDtype):
                cats = df_clean[cat_col].cat.categories
                cats = [c for c in cats if c in df_clean[cat_col].values]
            else:
                try:
                    cats = np.sort(cats)
                except:
                    pass
        else:
            cats = ["Data"]

        # Get Hues
        if hue:
            hues = df_clean[hue].unique()
            if isinstance(df_clean[hue].dtype, pd.CategoricalDtype):
                hues = df_clean[hue].cat.categories
                hues = [h for h in hues if h in df_clean[hue].values]
            else:
                try:
                    hues = np.sort(hues)
                except:
                    pass
        else:
            hues = [None]

        # Auto-detect dodge
        if dodge is None:
            if hue and cat_col and (hue == cat_col):
                dodge = False
            else:
                dodge = True

        # Resolve Palette
        if hue:
            if isinstance(palette, dict):
                color_map = palette
            else:
                if palette is None:
                    pal_list = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
                    pal_iter = itertools.cycle(pal_list)
                elif isinstance(palette, str):
                    # Check if it's a colormap
                    try:
                        cmap = mpl_colormaps[palette]
                        # Sample colors from colormap
                        if len(hues) > 1:
                            colors = cmap(np.linspace(0, 1, len(hues)))
                        else:
                            colors = [cmap(0.5)]
                        pal_iter = itertools.cycle(colors)
                    except (KeyError, ValueError):
                        # Not a colormap, treat as single color
                        pal_iter = itertools.cycle([palette])
                else:
                    pal_iter = itertools.cycle(palette)

                color_map = {h: next(pal_iter) for h in hues}
        else:
             # Legacy default: all blue if no palette, or cycle if palette
             pass

        # Build PlotItems
        total_width = 0.8
        n_hues = len(hues)
        if dodge:
            slot_width = total_width / n_hues
        else:
            slot_width = total_width

        for i_cat, cat in enumerate(cats):
            xtick_positions.append(i_cat)
            xtick_labels.append(str(cat))

            if cat_col:
                sub_df = df_clean[df_clean[cat_col] == cat]
            else:
                sub_df = df_clean

            for i_hue, h in enumerate(hues):
                if hue:
                    group_data = sub_df[sub_df[hue] == h]
                    label = f"{cat} - {h}"

                    if dodge:
                        offset = (i_hue - (n_hues - 1) / 2) * slot_width
                    else:
                        offset = 0

                    pos = i_cat + offset
                    col = color_map.get(h, "#4C78A8")
                else:
                    group_data = sub_df
                    label = str(cat)
                    pos = i_cat
                    offset = 0

                    if palette is None:
                        col = "#4C78A8"
                    elif isinstance(palette, str):
                        col = palette
                    elif isinstance(palette, dict):
                         col = palette.get(cat, "#4C78A8")
                    else:
                        pal_cycle = itertools.cycle(palette)
                        col = next(itertools.islice(pal_cycle, i_cat, i_cat + 1))

                vals = group_data[val_col].values

                if len(vals) == 0: continue

                plot_items.append({
                    'values': vals,
                    'position': pos,
                    'width': slot_width,
                    'color': col,
                    'label': label
                })

        if x_label is None: x_label = cat_col if vert else val_col
        if y_label is None: y_label = val_col if vert else cat_col

    else:
        # Branch B: Legacy Input Processing
        raw_plot_data = [] # List of (label, values)

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                vals = data[col].dropna().values
                raw_plot_data.append((str(col), vals))
        elif isinstance(data, dict):
            for label, values in data.items():
                vals = np.asarray(values)
                vals = vals[~np.isnan(vals)]
                raw_plot_data.append((str(label), vals))
        elif isinstance(data, (list, np.ndarray)):
            data_arr = np.asarray(data, dtype=object)
            is_multiple = False
            try:
                numeric_arr = np.asarray(data, dtype=float)
                if numeric_arr.ndim > 1:
                    is_multiple = True
                else:
                    is_multiple = False
            except (ValueError, TypeError):
                is_multiple = True

            if is_multiple:
                for i, d in enumerate(data):
                    vals = np.asarray(d, dtype=float)
                    vals = vals[~np.isnan(vals)]
                    raw_plot_data.append((str(i), vals))
            else:
                vals = np.asarray(data, dtype=float)
                vals = vals[~np.isnan(vals)]
                raw_plot_data.append(("Data", vals))
        else:
            raise ValueError("Unsupported data type")

        # Colors
        n_groups = len(raw_plot_data)

        # Determine effective labels for color lookup if palette is a dict
        effective_labels = [item[0] for item in raw_plot_data]

        # Check if user provided custom labels via x_label (vertical) or y_label (horizontal)
        labels_override = None
        if vert:
             if x_label is not None and isinstance(x_label, (list, tuple, np.ndarray)) and not isinstance(x_label, str):
                 if len(x_label) == n_groups:
                     labels_override = x_label
        else:
             if y_label is not None and isinstance(y_label, (list, tuple, np.ndarray)) and not isinstance(y_label, str):
                 if len(y_label) == n_groups:
                     labels_override = y_label

        if labels_override is not None:
            effective_labels = labels_override

        if palette is None:
            colors = ["#4C78A8"] * n_groups
        elif isinstance(palette, dict):
            colors = []
            for lbl in effective_labels:
                # Try exact match first, then string match
                if lbl in palette:
                    colors.append(palette[lbl])
                elif str(lbl) in palette:
                    colors.append(palette[str(lbl)])
                else:
                    colors.append("#4C78A8") # Fallback color
        elif isinstance(palette, str):
             # If single color string
            colors = [palette] * n_groups
        else:
            # List of colors
            colors = list(itertools.islice(itertools.cycle(palette), n_groups))

        for i, (label, vals) in enumerate(raw_plot_data):
            plot_items.append({
                'values': vals,
                'position': float(i),
                'width': 0.8,
                'color': colors[i],
                'label': label
            })
            xtick_positions.append(i)
            xtick_labels.append(label)

    # --- 2. Setup Figure/Axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- 4. Plotting Loop ---
    for item in plot_items:
        vals = item['values']
        pos = item['position']
        col = item['color']
        width = item['width']

        # A. Violin
        parts = ax.violinplot(vals, positions=[pos], widths=width,
                              showextrema=False, vert=vert)

        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(alpha_violin)

            # Clipping to half
            # get_paths()[0].vertices is (N, 2) array of (x, y)
            verts = pc.get_paths()[0].vertices

            if vert:
                # Vertical: x is dim 0. Center is 'pos'.
                mean_val = verts[:, 0].mean()
                verts[:, 0] = np.clip(verts[:, 0], None, mean_val)
            else:
                # Horizontal: y is dim 1. Center is 'pos'.
                mean_val = verts[:, 1].mean()
                verts[:, 1] = np.clip(verts[:, 1], mean_val, None)

        # B. Scatter (Rain)
        rng = np.random.default_rng(0)
        # Scale jitter and offset by width ratio relative to 0.8 (default)
        scale_factor = width / 0.8
        jitter_vals = rng.normal(loc=0, scale=0.04 * scale_factor, size=len(vals))
        offset = offset_scatter * scale_factor

        if vert:
            x_scatter = pos + offset + jitter_vals
            y_scatter = vals
        else:
            x_scatter = vals
            y_scatter = pos - offset + jitter_vals

        ax.scatter(x_scatter, y_scatter, color=col, alpha=alpha_scatter,
                   edgecolor="black", linewidth=linewidth_scatter, s=size_scatter)

        # C. Boxplot Elements (Median + IQR)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        if vert:
            ax.vlines(pos, q1, q3, color="k", linewidth=linewidth_boxplot, zorder=2)
            ax.scatter(pos, med, color="white", edgecolor="k", linewidth=2, s=size_median, zorder=3)
        else:
            ax.hlines(pos, q1, q3, color="k", linewidth=linewidth_boxplot, zorder=2)
            ax.scatter(med, pos, color="white", edgecolor="k", linewidth=2, s=size_median, zorder=3)

    # --- 5. Cosmetics ---
    if vert:
        if x_label is not None and isinstance(x_label, (list, tuple, np.ndarray)) and not isinstance(x_label, str):
            if len(x_label) == len(xtick_labels):
                xtick_labels = x_label
                x_label = None
            else:
                print(f"Warning: x_label list length ({len(x_label)}) does not match number of groups ({len(xtick_labels)}). Using as axis label.")
    else:
        if y_label is not None and isinstance(y_label, (list, tuple, np.ndarray)) and not isinstance(y_label, str):
            if len(y_label) == len(xtick_labels):
                xtick_labels = y_label
                y_label = None
            else:
                print(f"Warning: y_label list length ({len(y_label)}) does not match number of groups ({len(xtick_labels)}). Using as axis label.")

    if vert:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
    else:
        ax.set_yticks(xtick_positions)
        ax.set_yticklabels(xtick_labels)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return {"fig": fig, "axes": ax}


# Function to adjust colormaps
def colormap_maker(colors, positions=None, cmap_name=None):
    """
    Creates a new custom colormap from a list of colors and optional positions.

    If `cmap_name` is provided, the colormap is registered with Matplotlib and
    can be accessed via `plt.get_cmap(cmap_name)` or `cmap=cmap_name` in plotting functions.

    Parameters
    ----------
    colors : list
        A list of colors in any format accepted by Matplotlib (e.g., 'k', 'cyan',
        (1.0, 1.0, 1.0, 0.5), '#FF5733').
    positions : tuple or list of floats, optional
        A sequence of floats from 0.0 to 1.0 indicating the position of each color.
        If not provided, the colors will be equally spaced.
    cmap_name : str, optional
        The name to register the colormap under in Matplotlib. If None, the colormap
        is not registered.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The generated colormap object.

    Examples
    --------
    >>> # Create and register a 'cyberpunk' colormap
    >>> cmap = colormap_maker(['#08041c', '#390b5e', '#a2217c', '#f04e4c', '#fce205'], cmap_name='cyberpunk')
    >>> plt.imshow(data, cmap='cyberpunk')
    """
    if positions is not None:
        if len(positions) != len(colors):
            raise ValueError("The number of positions must match the number of colors.")
        if positions[0] != 0.0 or positions[-1] != 1.0:
            raise ValueError("Positions must start with 0.0 and end with 1.0.")
        if not all(positions[i] < positions[i+1] for i in range(len(positions)-1)):
            raise ValueError("Positions must be strictly monotonically increasing.")

        # LinearSegmentedColormap.from_list accepts a list of (value, color) tuples
        color_data = list(zip(positions, colors))
    else:
        color_data = colors

    # Generate a default name for internal creation if None is provided
    internal_name = cmap_name if cmap_name else "custom_cmap"

    cmap = LinearSegmentedColormap.from_list(internal_name, color_data)

    if cmap_name is not None:
        # Register the colormap
        # In newer Matplotlib versions, plt.colormaps.register or mpl.colormaps.register is preferred
        # but plt.register_cmap is the classic way. Let's use mpl_colormaps for compatibility.
        try:
            mpl_colormaps.register(cmap, name=cmap_name, force=True)
        except AttributeError:
            # Fallback for older Matplotlib versions
            plt.register_cmap(name=cmap_name, cmap=cmap)

    return cmap

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

def plotly_scatter_3d(data=None, x=None, y=None, z=None, color=None, labels=None,
                      marker_size=5, opacity=0.8, colorscale='Purples',
                      initial_view=None, **kwargs):
    """
    Creates an interactive 3D scatter plot using Plotly.

    Parameters
    ----------
    data : pd.DataFrame or dict, optional
        Data source. If provided, `x`, `y`, `z`, `color`, and `labels` can be column names/keys.
    x, y, z : str or array-like
        Coordinates. Can be column names (if `data` is provided) or arrays.
    color : str or array-like, optional
        Color values. Can be a column name or an array.
    labels : str or array-like, optional
        Labels for hover text. Can be a column name or an array.
    marker_size : float or array-like, default 5
        Size of the markers.
    opacity : float, default 0.8
        Opacity of the markers.
    colorscale : str, default 'Purples'
        Name of the colorscale to use.
    initial_view : dict, optional
        Dictionary to set the initial camera view (e.g., `dict(eye=dict(x=1.5, y=1.5, z=1.5))`).
    **kwargs :
        Additional keyword arguments passed to `go.Scatter3d`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Plotly is required for this function. Install it with `pip install .[plotting]`") from e

    # Helper to resolve input to array
    def resolve_input(val, data_obj):
        if val is None:
            return None
        if isinstance(val, str) and data_obj is not None:
            if isinstance(data_obj, pd.DataFrame):
                return data_obj[val]
            elif isinstance(data_obj, dict):
                return data_obj[val]
        return val

    # Resolve inputs
    x_data = resolve_input(x, data)
    y_data = resolve_input(y, data)
    z_data = resolve_input(z, data)
    c_data = resolve_input(color, data)
    l_data = resolve_input(labels, data)

    if x_data is None or y_data is None or z_data is None:
        raise ValueError("x, y, and z coordinates must be provided.")

    # Prepare marker dict
    marker = dict(
        size=marker_size,
        opacity=opacity,
        colorscale=colorscale,
    )

    # Handle color
    if c_data is not None:
        if not pd.api.types.is_numeric_dtype(c_data):
             try:
                # Ensure it's a Series or convert
                c_series = pd.Series(c_data)
                c_codes = c_series.astype('category').cat.codes
                marker['color'] = c_codes
             except Exception as e:
                print(f"Warning: Failed to convert non-numeric color data to codes: {e}")
                marker['color'] = c_data
        else:
            marker['color'] = c_data

        marker['showscale'] = True
        marker['colorbar'] = dict(title='Value')

    # Construct hover template
    hovertemplate = (
        '<b>X:</b> %{x:.2f}<br>'
        '<b>Y:</b> %{y:.2f}<br>'
        '<b>Z:</b> %{z:.2f}<br>'
    )
    if l_data is not None:
        hovertemplate = '<b>Label:</b> %{text}<br>' + hovertemplate

    if c_data is not None:
        hovertemplate += '<b>Color:</b> %{marker.color:.2f}<br>'

    hovertemplate += '<extra></extra>'

    # Create Scatter3d trace
    trace = go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=marker,
        text=l_data,
        hovertemplate=hovertemplate,
        **kwargs
    )

    fig = go.Figure(data=[trace])

    # Update layout
    scene = dict(
        xaxis_title=x if isinstance(x, str) else 'X',
        yaxis_title=y if isinstance(y, str) else 'Y',
        zaxis_title=z if isinstance(z, str) else 'Z',
        aspectmode='data'
    )

    if initial_view:
        scene['camera'] = initial_view

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def plotly_scatter_3d_from_adata_obsm(adata, obsm_key, color_key=None, label_key=None, **kwargs):
    """
    Creates an interactive 3D scatter plot from an AnnData object's obsm slot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obsm_key : str
        Key for `adata.obsm` to use as coordinates (e.g., 'X_umap', 'X_pca').
        Expects at least 3 dimensions.
    color_key : str, optional
        Key for `adata.obs` to use for coloring.
    label_key : str, optional
        Key for `adata.obs` to use for labels.
    **kwargs :
        Additional keyword arguments passed to `plotly_scatter_3d`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    if obsm_key not in adata.obsm:
        raise ValueError(f"'{obsm_key}' not found in adata.obsm.")

    coords = adata.obsm[obsm_key]

    if coords.shape[1] < 3:
        # Warn or handle gracefully? User asked for 3D.
        # We'll pad with zeros if 2D, but warn.
        print(f"Warning: '{obsm_key}' has only {coords.shape[1]} dimensions. Padding with zeros for Z-axis.")
        z = np.zeros(coords.shape[0])
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

    # Extract color and labels
    color = None
    if color_key:
        if color_key in adata.obs:
            color = adata.obs[color_key]
        else:
            print(f"Warning: '{color_key}' not found in adata.obs.")

    labels = None
    if label_key:
        if label_key in adata.obs:
            labels = adata.obs[label_key]
        else:
            print(f"Warning: '{label_key}' not found in adata.obs.")

    # Set default axis titles if not provided in kwargs (via passing x,y,z names)
    # We pass arrays directly, so plotly_scatter_3d won't see names unless we pass them or update layout.
    # To get nice axis titles, we can manually update the figure afterwards or update the scene dict inside.
    # But plotly_scatter_3d infers titles from string args. Here we pass arrays.

    fig = plotly_scatter_3d(x=x, y=y, z=z, color=color, labels=labels, **kwargs)

    # Update axis titles to reflect the keys
    fig.update_layout(scene=dict(
        xaxis_title=f"{obsm_key}_0",
        yaxis_title=f"{obsm_key}_1",
        zaxis_title=f"{obsm_key}_2" if coords.shape[1] >= 3 else "Zero",
    ))

    return fig

def get_nice_number(value):
    """
    Rounds a number to a 'nice' round number for legends.
    Examples: 4343 -> 4000, 10858 -> 10000, 21717 -> 20000
    """
    if value == 0: return 0

    # Calculate magnitude
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)

    # Round fraction to nice intervals (1, 2, 5, 10)
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return int(nice_fraction * (10 ** exponent))


def savefig_svg(filename, bgnd_color=(1, 1, 1, 0.8), bbox_inches='tight', dpi=300, pad_inches=0.1, **kwargs):
    """
    Saves the currently active matplotlib figure as an SVG file.

    Parameters
    ----------
    filename : str or Path
        The path where the SVG will be saved. The '.svg' extension is appended if missing.
    bgnd_color : color, default (1, 1, 1, 0.8)
        The background color (facecolor) of the saved figure.
    bbox_inches : str or Bbox, default 'tight'
        Bounding box in inches: only the given portion of the figure is saved.
    dpi : float, default 300
        The resolution in dots per inch for rasterized elements.
    pad_inches : float, default 0.1
        Amount of padding around the figure when bbox_inches is 'tight'.
    **kwargs :
        Additional keyword arguments passed directly to `plt.savefig`.
    """
    fig = plt.gcf()

    # Extract figure title for metadata
    title = fig._suptitle.get_text() if fig._suptitle else str(filename)

    # Initialize metadata with predefined options
    metadata = kwargs.pop('metadata', {})
    metadata.setdefault('Creator', 'eigenp')
    metadata.setdefault('Date', datetime.datetime.now().isoformat())
    metadata.setdefault('Title', title)

    # Make sure we add '.svg' if it's not present
    filename_str = str(filename)
    if not filename_str.lower().endswith('.svg'):
        filename_str += '.svg'

    # Call savefig on the current figure
    fig.savefig(
        filename_str,
        format='svg',
        facecolor=bgnd_color,
        bbox_inches=bbox_inches,
        dpi=dpi,
        pad_inches=pad_inches,
        metadata=metadata,
        **kwargs
    )
    print(f"Saved figure to {filename_str}")
