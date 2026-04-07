# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anywidget",
#     "traitlets",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
# ]
# ///

import pathlib
import warnings
import anywidget
import traitlets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import importlib.resources as ir
from matplotlib import gridspec

from skimage.transform import resize
from matplotlib.colors import PowerNorm, to_rgb, LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


def is_colormap(c):
    """
    Checks if a string is a valid matplotlib colormap name.
    """
    if not isinstance(c, str):
        return False
    try:
        plt.get_cmap(c)
        return True
    except ValueError:
        return False

def resolve_color(c):
    """
    Attempts to resolve a string to a valid matplotlib color.
    If it is a valid colormap name, it returns the final color (at 1.0) of that colormap.
    """
    if not isinstance(c, str):
        return c

    try:
        # Check if it's already a valid color name or hex
        to_rgb(c)
        return c
    except ValueError:
        pass

    if is_colormap(c):
        cmap = plt.get_cmap(c)
        return mcolors.to_hex(cmap(1.0)[:3])  # Get hex of final color

    return c

def _norm(arr, symmetric=False, eps=1e-12, dtype=np.float32):
    a = arr.astype(dtype, copy=False)
    if symmetric:
        d = max(eps, float(np.abs(a).max()))
    else:
        d = max(eps, float(a.max()))
    return a / d


# Copyright tnia 2021 - BSD License
def show_zyx_slice(image_to_show, x, y, z, sxy=None, sz=None,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, use_plt=True, opacity=None):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        x (int): x position of slice
        y (int): y position of slice
        z (int): z position of slice
        sxy (float, optional): xy pixel size of 3D. Defaults to None.
        sz (float, optional): z pixel size of 3D. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """

    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    return show_zyx(slice_xy, slice_xz, slice_zy, sxy, sz, figsize, colormap, vmax = vmax, vmin = vmin, gamma = gamma, use_plt = use_plt, opacity = opacity)

# Copyright tnia 2021 - BSD License
def show_zyx_max(image_to_show, sxy=None, sz=None,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None):
    """ plots max xy, xz, and zy projections of a 3D image

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size. Defaults to None.
        sz (float, optional): z pixel size. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors

    return show_zyx_projection(image_to_show, sxy, sz, figsize, np.max, colormap, vmax=vmax, vmin = vmin, gamma = gamma, colors = colors, opacity = opacity)


def show_zyx_projection(image_to_show, sxy=None, sz=None,figsize=(10,10), projector=np.max, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to None.
        sz (float, optional): z pixel size of 3D. Defaults to None.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors
    projection_y = projector(image_to_show, axis=1)
    projection_x = np.flip(np.rot90(projector(image_to_show, axis=2), 1), 0)
    projection_z = projector(image_to_show, axis=0)

    return show_zyx(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmax=vmax, vmin = vmin, gamma = gamma, colors = colors, opacity = opacity)

# Copyright tnia 2021 - BSD License
def show_zyx(xy, xz, zy, sxy=None, sz=None,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, use_plt=True, colors = None, opacity = None, subplot_bg=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to None (treats as 1).
        sz (float, optional): z pixel size of 3D. Defaults to None (treats as 1).
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    Returns:
        [type]: [description]
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors
    both_given = sxy is not None and sz is not None
    if sxy is None:
        sxy = 1
    if sz is None:
        sz = 1

    if isinstance(xy,list):
        MULTI_CHANNEL = True
        has_colormap = False
        if colormap is not None:
            has_colormap = any(is_colormap(c) for c in colormap)
        else:
            has_colormap = False

        if has_colormap:
            xy, xz, zy = create_multichannel_rgb_cmap(xy, xz, zy, vmin=vmin, vmax=vmax, gamma=gamma, colormap=colormap, opacity=opacity)
        else:
            xy, xz, zy = create_multichannel_rgb(xy, xz, zy, vmin = vmin, vmax=vmax, gamma=gamma, colormap=colormap, opacity=opacity)

        colormap = None

        # Set those back to default bcs they are dealt with in the RGB function
        vmin, vmax, gamma = None, None, 1
        # Set opacity back to None, it is applied by create_multichannel_rgb
        opacity = None
    else:
        # In single channel, ensure these are floats/None, not lists
        if isinstance(opacity, list): opacity = opacity[0]
        if isinstance(vmin, list): vmin = vmin[0]
        if isinstance(vmax, list): vmax = vmax[0]

        # If colormap is provided as a list with one item, unpack it
        if isinstance(colormap, list) and len(colormap) == 1:
            colormap = colormap[0]

        if colormap is not None:
            c = colormap
            if isinstance(c, str):
                try:
                    plt.get_cmap(c)
                except ValueError:
                    resolved = resolve_color(c)
                    colormap = black_to(resolved)
            else:
                resolved = resolve_color(c)
                colormap = black_to(resolved)


    if use_plt:
        fig=plt.figure(figsize=figsize, constrained_layout=False)
    else:
        fig = Figure(figsize=figsize, constrained_layout=False)


    xdim = xy.shape[1]
    ydim = xy.shape[0]
    zdim = xz.shape[0]

    z_xy_ratio=1

    if sxy!=sz:
        z_xy_ratio=sz/sxy


    # compute the same-gap factor
    if figsize is not None:
        figW, figH = figsize
        hspace_factor = figW / figH
    else:
        figH = 10
        hspace_factor = 1.0

    spec=gridspec.GridSpec(ncols=2, nrows=2,
                           height_ratios=[ydim,zdim*z_xy_ratio],
                           width_ratios=[xdim,zdim*z_xy_ratio],
                           hspace=.01 * hspace_factor,
                           wspace=.01,
                           figure = fig)

    ax0=fig.add_subplot(spec[0])
    ax1=fig.add_subplot(spec[1])
    ax2=fig.add_subplot(spec[2])
    ax3=fig.add_subplot(spec[3])

    if gamma == 1:
        ax0.imshow(xy, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0], interpolation = 'nearest', alpha=opacity)
        ax1.imshow(zy, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0], interpolation = 'nearest', alpha=opacity)
        ax2.imshow(xz, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0], interpolation = 'nearest', alpha=opacity)
    else:
        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
        ax0.imshow(xy, cmap = colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0], interpolation = 'nearest', alpha=opacity)
        ax1.imshow(zy, cmap = colormap, norm=norm, extent=[0,zdim*sz,ydim*sxy,0], interpolation = 'nearest', alpha=opacity)
        ax2.imshow(xz, cmap = colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0], interpolation = 'nearest', alpha=opacity)

    ### Axes and titles
    # ax0.set_title('xy')
    # ax1.set_title('zy')
    # ax2.set_title('xz')

    # Remove in-between axes ticks
    for i, ax in enumerate([ax0,ax1,ax2, ax3]):
        if i < 3 and subplot_bg is not None:
            ax.set_facecolor(subplot_bg)
        else:
            ax.patch.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # ax0.xaxis.set_ticklabels([])
    # ax1.yaxis.set_ticklabels([])

    fig.patch.set_alpha(0.0) # set transparent bgnd

    # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Add scale bar
    width_um = xdim * sxy
    target = width_um * 0.2

    # a small utility to pick the largest “nice” number ≤ target
    def nice_length(x):
        # get exponent
        exp = np.floor(np.log10(x))
        # candidates: 1, 2, 5 times 10^exp
        for m in [5,2,1]:
            val = m * 10**exp
            if val <= x:
                return val
        # if nothing smaller (very small x), just return x itself
        return x

    bar_um = nice_length(target)

    # Convert back to pixels
    bar_pix = bar_um / sxy
    bar_frac = bar_pix / xdim    # fraction of the full width

    # pick fontsize
    fig_h_in = figsize[1] if figsize is not None else 10
    fontsize_pt = max(8, min(24, fig_h_in * 72 * 0.03))

    ### Draw
    # center the bar at (x=0.5), y=0.5 in ax3’s normalized coordinates:
    x0 = 0.5 - bar_frac/2
    x1 = 0.5 + bar_frac/2
    y  = 0.5

    ax3.hlines(y, x0, x1, transform=ax3.transAxes,
               linewidth=2, color='gray')

    if both_given:
        text_label = f"{int(bar_um)} µm"
    else:
        text_label = "`sxy` , `sz`"

    ax3.text(0.5, y - 0.1, text_label,
             transform=ax3.transAxes,
             ha='center', va='top',
             color='gray',
             fontsize=fontsize_pt)

    return fig



### New function
def show_zyx_max_slabs(image_to_show, x = [0,1], y = [0,1], z = [0,1], sxy=None, sz=None,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None):
    """ plots max xy, xz, and zy projections of a 3D image SLABS (slice intervals)

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        x: slices for x in format [x_1, x_2] where values are integers, to be passed as slice(x_1, x_2, None)
        y: slices for y in format [y_1, y_2] where values are integers
        z: slices for z in format [z_1, z_2] where values are integers
        sxy (float, optional): xy pixel size of 3D. Defaults to None.
        sz (float, optional): z pixel size of 3D. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors
    ### Coerce into integers for slices
    x_ = [int(i) for i in x]
    y_ = [int(i) for i in y]
    z_ = [int(i) for i in z]

    x_slices = slice(*x_)
    y_slices = slice(*y_)
    z_slices = slice(*z_)

    return show_zyx_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy, sz, figsize, np.max, colormap, vmax = vmax, vmin = vmin, gamma = gamma, colors = colors, opacity = opacity)


### New function
def show_zyx_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy=None, sz=None,figsize=(10,10), projector=np.max, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to None.
        sz (float, optional): z pixel size of 3D. Defaults to None.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors

    if isinstance(image_to_show, list):
        images_to_show_list = image_to_show
        projection_y = []
        projection_x = []
        projection_z = []
        for img in images_to_show_list:
            # Slicing creates a view. Some projectors might make copies. np.max over an axis
            # might allocate a temporary buffer if the memory is not contiguous in that axis.
            projection_y.append(projector(img[:, y_slices, :], axis=1))
            projection_x.append(np.flip(np.rot90(projector(img[:, :, x_slices], axis=2), 1), 0))
            projection_z.append(projector(img[z_slices, :, :], axis=0))
    else:
        projection_y = projector(image_to_show[:, y_slices, :], axis=1)
        projection_x = np.flip(np.rot90(projector(image_to_show[:, :, x_slices], axis=2), 1), 0)
        projection_z = projector(image_to_show[z_slices, :, :], axis=0)

    return show_zyx(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmax = vmax, vmin = vmin, gamma = gamma, colors = colors, opacity = opacity)




### New Function
# def create_multichannel_rgb(xy_list, xz_list, zy_list, vmin = None, vmax = None, gamma = 1, colors = None):
#     """
#     Display an interactive widget to explore a 3D image by showing a slice in the x, y, and z directions.

#     Requires ipywidgets to be installed.

#     Parameters
#     ----------
#     xy_list, xz_list, zy_list : lists of images (len of list is number of channels)
#     vmax : float
#         maximum value to use for the PowerNorm
#     gamma : float
#         gamma value to use for the PowerNorm
#     colors : list of strs
#         one color per channel
#     """

#     assert isinstance(xy_list,list)

#     num_channels = len(xy_list)

#     if gamma == 1:
#         gamma = [1] * num_channels
#     if vmax is None:
#         vmax = [1] * num_channels
#     if vmin is None:
#         vmin = [0] * num_channels

#     if colors is None:
#         colors = ['magenta', 'cyan', 'yellow', 'green']
#         colors = colors[0:num_channels]
#     # Convert color names or tuples to RGB
#     color_map = [to_rgb(color) for color in colors]


#     # # Initialize RGB arrays for each orientation
#     # xy_rgb = np.zeros(xy_list[0].shape + (3,))
#     # xz_rgb = np.zeros(xz_list[0].shape + (3,))
#     # zy_rgb = np.zeros(zy_list[0].shape + (3,))

#     # # Apply PowerNorm per channel
#     # for idx_i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
#     #     eps = 1e-12
#     #     xy = xy / max(eps, float(np.max(xy)))
#     #     xz = xz / max(eps, float(np.max(xz)))
#     #     zy = zy / max(eps, float(np.max(zy)))
#     #     norm = PowerNorm(gamma=gamma[idx_i], vmin=vmin[idx_i], vmax=vmax[idx_i], clip = True)
#     #     xy_norm, xz_norm, zy_norm = norm(xy), norm(xz), norm(zy)  # manually applying norm to the image data
#     #     # xy_list[idx_i], xz_list[idx_i], zy_list[idx_i] = [idx_i]

#     #     # Combine channels into RGB using color weights
#     #     xy_rgb += np.outer(xy_norm.flatten(), color_map[idx_i]).reshape(xy_norm.shape + (3,))
#     #     xz_rgb += np.outer(xz_norm.flatten(), color_map[idx_i]).reshape(xz_norm.shape + (3,))
#     #     zy_rgb += np.outer(zy_norm.flatten(), color_map[idx_i]).reshape(zy_norm.shape + (3,))



def create_multichannel_rgb(
    xy_list, xz_list, zy_list,
    vmin=None, vmax=None, gamma=1, colormap=None, colors=None, opacity=None,
    blend='add',        # 'add' | 'screen' | 'max'
    soft_clip=True,     # only used for blend='add'
    eps=1e-12,
):
    """
    Compose multi-channel XY/XZ/ZY into RGB with per-channel normalization.

    Parameters
    ----------
    xy_list, xz_list, zy_list : list[np.ndarray]
        One 2D array per channel for each orientation.
    vmin, vmax : float | list[float] | None
        Per-channel input range. If None, auto-computed from data (per-channel,
        across XY/XZ/ZY). Scalars are broadcast to all channels.
    gamma : float | list[float]
        Power gamma applied AFTER linear [0,1] normalization (1 = linear).
    colors : list[str | tuple]
        One color per channel (default: ['magenta','cyan','yellow','green'][:n]).
    opacity : float | list[float] | None
        Opacity multiplier per channel (1 = full opacity).
    blend : str
        'add' (default), 'screen', or 'max'.
    soft_clip : bool
        For 'add' mode, compress values >1 instead of hard clipping.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors

    assert isinstance(xy_list, list) and isinstance(xz_list, list) and isinstance(zy_list, list)
    n = len(xy_list)
    assert len(xz_list) == n and len(zy_list) == n, "xy/xz/zy must have same number of channels"

    Hxy, Wxy = xy_list[0].shape
    Hxz, Wxz = xz_list[0].shape
    Hzy, Wzy = zy_list[0].shape

    # Prepare outputs
    xy_rgb = np.zeros((Hxy, Wxy, 3), dtype=np.float32)
    xz_rgb = np.zeros((Hxz, Wxz, 3), dtype=np.float32)
    zy_rgb = np.zeros((Hzy, Wzy, 3), dtype=np.float32)

    # Broadcast params
    gammas = (list(gamma) if isinstance(gamma, (list, tuple)) else [gamma] * n)
    opacities = (list(opacity) if isinstance(opacity, (list, tuple)) else [opacity if opacity is not None else 1.0] * n)

    if colormap is None:
        if n == 1:
            colormap = ['white']
        else:
            defaults = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
            colormap = [defaults[i % len(defaults)] for i in range(n)]
    color_map = [np.asarray(to_rgb(resolve_color(c)), dtype=np.float32) for c in colormap]

    # Determine per-channel vmin/vmax if not provided
    # vmin/vmax can be None, or a list containing floats and/or Nones
    if vmin is None:
        vmins = [0.0] * n
    else:
        vmins = list(vmin) if isinstance(vmin, (list, tuple)) else [vmin] * n

    if vmax is None:
        vmaxs = [None] * n
    else:
        vmaxs = list(vmax) if isinstance(vmax, (list, tuple)) else [vmax] * n

    for i in range(n):
        if vmins[i] is None:
            vmins[i] = 0.0
        else:
            vmins[i] = float(vmins[i])

        if vmaxs[i] is None:
            # max over 2D slices instead of concatenating
            m_xy = float(np.max(xy_list[i]))
            m_xz = float(np.max(xz_list[i]))
            m_zy = float(np.max(zy_list[i]))
            vmaxs[i] = float(max(m_xy, m_xz, m_zy))
        else:
            vmaxs[i] = float(vmaxs[i])

    # Sanitize: ensure vmax > vmin
    for i in range(n):
        if not np.isfinite(vmins[i]): vmins[i] = 0.0
        if not np.isfinite(vmaxs[i]): vmaxs[i] = vmins[i] + 1.0
        if vmaxs[i] <= vmins[i] + eps:
            vmaxs[i] = vmins[i] + 1.0  # avoid zero range

    # Choose blending accumulators
    if blend == 'screen':
        xy_acc = np.ones_like(xy_rgb)
        xz_acc = np.ones_like(xz_rgb)
        zy_acc = np.ones_like(zy_rgb)
    else:
        xy_acc = xy_rgb
        xz_acc = xz_rgb
        zy_acc = zy_rgb

    # Helpers
    def _norm(a, lo, hi, g):
        # linear normalize to [0,1] then gamma
        out = (a.astype(np.float32, copy=False) - lo) / max(hi - lo, eps)
        out = np.clip(out, 0.0, 1.0, out=out)
        if g != 1:
            out = np.power(out, g, out=out)
        return out

    # Per-channel accumulate
    for i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
        c = color_map[i]  # (3,)
        g = gammas[i]
        o = opacities[i]
        lo, hi = vmins[i], vmaxs[i]

        # multiply by color * opacity
        c_o = (c * o).astype(np.float32)

        xy_n = _norm(xy, lo, hi, g)[..., None] * c_o  # (H,W,3)
        xz_n = _norm(xz, lo, hi, g)[..., None] * c_o
        zy_n = _norm(zy, lo, hi, g)[..., None] * c_o

        if blend == 'screen':
            xy_acc *= (1.0 - xy_n)
            xz_acc *= (1.0 - xz_n)
            zy_acc *= (1.0 - zy_n)
        elif blend == 'max':
            xy_acc = np.maximum(xy_acc, xy_n)
            xz_acc = np.maximum(xz_acc, xz_n)
            zy_acc = np.maximum(zy_acc, zy_n)
        else:  # 'add'
            xy_acc += xy_n
            xz_acc += xz_n
            zy_acc += zy_n

    # Finalize per blend
    if blend == 'screen':
        xy_rgb = 1.0 - xy_acc
        xz_rgb = 1.0 - xz_acc
        zy_rgb = 1.0 - zy_acc
    else:
        xy_rgb = xy_acc
        xz_rgb = xz_acc
        zy_rgb = zy_acc

        if blend == 'add':
            if soft_clip:
                # compress highlights smoothly instead of hard clipping
                # scale by max component per-pixel if it exceeds 1
                for rgb in (xy_rgb, xz_rgb, zy_rgb):
                    m = rgb.max(axis=-1, keepdims=True)
                    scale = np.maximum(1.0, m)
                    rgb /= scale
            # ensure display-safe range
            xy_rgb = np.clip(xy_rgb, 0.0, 1.0)
            xz_rgb = np.clip(xz_rgb, 0.0, 1.0)
            zy_rgb = np.clip(zy_rgb, 0.0, 1.0)

    return xy_rgb, xz_rgb, zy_rgb

    # # return show_zyx(xy_rgb, xz_rgb, zy_rgb, vmin = None, vmax=None, gamma = 1, use_plt=True)
    # return xy_rgb, xz_rgb, zy_rgb

def create_multichannel_rgb_cmap(
    xy_list, xz_list, zy_list,
    vmin=None, vmax=None, gamma=1, colormap=None, colors=None, opacity=None,
    blend='max',        # 'add' | 'screen' | 'max'
    soft_clip=True,     # only used for blend='add'
    eps=1e-12,
):
    """
    Compose multi-channel XY/XZ/ZY into RGB with per-channel normalization using full colormaps.
    """
    if colors is not None:
        warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
        if colormap is None:
            colormap = colors
    assert isinstance(xy_list, list) and isinstance(xz_list, list) and isinstance(zy_list, list)
    n = len(xy_list)
    assert len(xz_list) == n and len(zy_list) == n, "xy/xz/zy must have same number of channels"

    Hxy, Wxy = xy_list[0].shape
    Hxz, Wxz = xz_list[0].shape
    Hzy, Wzy = zy_list[0].shape

    # Prepare outputs
    xy_rgb = np.zeros((Hxy, Wxy, 3), dtype=np.float32)
    xz_rgb = np.zeros((Hxz, Wxz, 3), dtype=np.float32)
    zy_rgb = np.zeros((Hzy, Wzy, 3), dtype=np.float32)

    # Broadcast params
    gammas = (list(gamma) if isinstance(gamma, (list, tuple)) else [gamma] * n)
    opacities = (list(opacity) if isinstance(opacity, (list, tuple)) else [opacity if opacity is not None else 1.0] * n)

    if colormap is None:
        if n == 1:
            colormap = ['white']
        else:
            defaults = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
            colormap = [defaults[i % len(defaults)] for i in range(n)]

    cmap_list = []
    for c in colormap:
        if is_colormap(c):
            cmap_list.append(plt.get_cmap(c))
        else:
            cmap_list.append(black_to(resolve_color(c)))

    # Determine per-channel vmin/vmax if not provided
    if vmin is None:
        vmins = [0.0] * n
    else:
        vmins = list(vmin) if isinstance(vmin, (list, tuple)) else [vmin] * n

    if vmax is None:
        vmaxs = [None] * n
    else:
        vmaxs = list(vmax) if isinstance(vmax, (list, tuple)) else [vmax] * n

    for i in range(n):
        if vmins[i] is None:
            vmins[i] = 0.0
        else:
            vmins[i] = float(vmins[i])

        if vmaxs[i] is None:
            m_xy = float(np.max(xy_list[i]))
            m_xz = float(np.max(xz_list[i]))
            m_zy = float(np.max(zy_list[i]))
            vmaxs[i] = float(max(m_xy, m_xz, m_zy))
        else:
            vmaxs[i] = float(vmaxs[i])

    # Sanitize: ensure vmax > vmin
    for i in range(n):
        if not np.isfinite(vmins[i]): vmins[i] = 0.0
        if not np.isfinite(vmaxs[i]): vmaxs[i] = vmins[i] + 1.0
        if vmaxs[i] <= vmins[i] + eps:
            vmaxs[i] = vmins[i] + 1.0  # avoid zero range

    # Choose blending accumulators
    if blend == 'screen':
        xy_acc = np.ones_like(xy_rgb)
        xz_acc = np.ones_like(xz_rgb)
        zy_acc = np.ones_like(zy_rgb)
    else:
        xy_acc = xy_rgb
        xz_acc = xz_rgb
        zy_acc = zy_rgb

    # Helpers
    def _norm(a, lo, hi, g):
        out = (a.astype(np.float32, copy=False) - lo) / max(hi - lo, eps)
        out = np.clip(out, 0.0, 1.0, out=out)
        if g != 1:
            out = np.power(out, g, out=out)
        return out

    # Per-channel accumulate
    for i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
        cmap = cmap_list[i]
        g = gammas[i]
        o = opacities[i]
        lo, hi = vmins[i], vmaxs[i]

        # Apply colormap to normalized array
        xy_n = (cmap(_norm(xy, lo, hi, g), bytes=True)[..., :3].astype(np.float32) / 255.0) * o
        xz_n = (cmap(_norm(xz, lo, hi, g), bytes=True)[..., :3].astype(np.float32) / 255.0) * o
        zy_n = (cmap(_norm(zy, lo, hi, g), bytes=True)[..., :3].astype(np.float32) / 255.0) * o

        if blend == 'screen':
            xy_acc *= (1.0 - xy_n)
            xz_acc *= (1.0 - xz_n)
            zy_acc *= (1.0 - zy_n)
        elif blend == 'max':
            xy_acc = np.maximum(xy_acc, xy_n)
            xz_acc = np.maximum(xz_acc, xz_n)
            zy_acc = np.maximum(zy_acc, zy_n)
        else:  # 'add'
            xy_acc += xy_n
            xz_acc += xz_n
            zy_acc += zy_n

    # Finalize per blend
    if blend == 'screen':
        xy_rgb = 1.0 - xy_acc
        xz_rgb = 1.0 - xz_acc
        zy_rgb = 1.0 - zy_acc
    else:
        xy_rgb = xy_acc
        xz_rgb = xz_acc
        zy_rgb = zy_acc

        if blend == 'add':
            if soft_clip:
                for rgb in (xy_rgb, xz_rgb, zy_rgb):
                    m = rgb.max(axis=-1, keepdims=True)
                    scale = np.maximum(1.0, m)
                    rgb /= scale
            xy_rgb = np.clip(xy_rgb, 0.0, 1.0)
            xz_rgb = np.clip(xz_rgb, 0.0, 1.0)
            zy_rgb = np.clip(zy_rgb, 0.0, 1.0)

    return xy_rgb, xz_rgb, zy_rgb

def black_to(color):
    """Return a black→color LinearSegmentedColormap."""
    rgb = to_rgb(color)
    return LinearSegmentedColormap.from_list(f"black_to_{color}", [(0,0,0), rgb])


def blend_colors(intensities, base_colors, vmin=None, vmax=None, gamma=1, soft_clip=True):
    """
    Blend multiple channels of intensities into RGB.

    intensities : (N, C) array
        Values per point per channel.
    base_colors : list of str or rgb tuples
        Colors per channel.

    Notes:
        - Multi-channel visualization widgets support rendering with colormaps by distinguishing strings
          via `is_colormap()`. When active, intensities map directly using `cmap(norm)[:, :3]` rather
          than failing through `matplotlib.colors.to_rgb()`.
        - The `colors` parameter in top-level functions is deprecated in favor of `colormap`. To support
          multi-channel arrays, `colormap` accepts a list of colormaps. To prevent `TypeError: unhashable type: 'list'`
          from Matplotlib's `imshow`, functions explicitly unset `colormap` after mapping channels into a pre-rendered RGB array.
        - The default colormap for interactive plotters (slice, scatter, annotator) when no colormap is provided
          is `['white']` for single-channel data, and cycles through `['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']`
          for multi-channel data.
    """
    import numpy as np
    from matplotlib.colors import to_rgb

    N, C = intensities.shape
    colors = np.zeros((N, 3), dtype=float)

    vmin = np.zeros(C) if vmin is None else np.broadcast_to(vmin, (C,))
    vmax = np.ones(C)  if vmax is None else np.broadcast_to(vmax, (C,))
    gammas = np.ones(C) if np.isscalar(gamma) else np.asarray(gamma)

    for c in range(C):
        arr = intensities[:, c].astype(float)
        vmin_c = np.nanmin(arr) if vmin[c] is None else vmin[c]
        vmax_c = np.nanmax(arr) if vmax[c] is None else vmax[c]
        norm = (arr - vmin_c) / max(1e-9, vmax_c - vmin_c)
        norm = np.clip(norm, 0, 1)
        if gammas[c] != 1:
            norm = norm**gammas[c]

        c_name = base_colors[c]
        if is_colormap(c_name):
            cmap = plt.get_cmap(c_name)
            rgb = cmap(norm)[:, :3]
            colors += rgb
        else:
            resolved_c = resolve_color(c_name)
            rgb = np.asarray(to_rgb(resolved_c))
            colors += norm[:, None] * rgb

    if soft_clip:
        maxval = colors.max(axis=1, keepdims=True)
        scale = np.maximum(1.0, maxval)
        colors = colors / scale

    return np.clip(colors, 0, 1)





def compute_histogram(arr, bins=128):
    """
    Computes a 1D histogram for array values.

    To enhance the visibility of sparse/heavy-tailed intensity distributions in interactive
    widgets, the returned bin counts are transformed into log frequencies (using log1p).
    """
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    arr_clean = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if arr_clean.size == 0:
        return {'counts': [], 'bin_edges': []}

    # Determine absolute range based on dtype
    range_val = None
    if arr.dtype == np.uint8:
        range_val = (0, 255)
    elif arr.dtype == np.uint16:
        range_val = (0, 65535)
    elif arr.dtype == bool:
        range_val = (0, 1)

    if range_val is not None:
        counts, bin_edges = np.histogram(arr_clean, bins=bins, range=range_val)
    else:
        counts, bin_edges = np.histogram(arr_clean, bins=bins)

    # Use log frequencies to enhance visibility of tails in widgets
    counts = np.log1p(counts)

    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}

class TNIAWidgetBase(anywidget.AnyWidget):
    # _esm = pathlib.Path(__file__).parent / "tnia_plotting_anywidgets.js"
    # _css = pathlib.Path(__file__).parent / "tnia_plotting_anywidgets.css"
    _esm = ir.files("eigenp_utils").joinpath("tnia_plotting_anywidgets.js")
    _css = ir.files("eigenp_utils").joinpath("tnia_plotting_anywidgets.css")


    # Data traits
    image_data = traitlets.Unicode().tag(sync=True)

    # Sliders
    x_s = traitlets.Int(0).tag(sync=True)
    y_s = traitlets.Int(0).tag(sync=True)
    z_s = traitlets.Int(0).tag(sync=True)

    x_t = traitlets.Int(1).tag(sync=True)
    y_t = traitlets.Int(1).tag(sync=True)
    z_t = traitlets.Int(1).tag(sync=True)

    sxy = traitlets.Float(1.0).tag(sync=True)
    sz = traitlets.Float(1.0).tag(sync=True)

    # Bounds for sliders (computed)
    x_min_pos = traitlets.Int(0).tag(sync=True)
    x_max_pos = traitlets.Int(100).tag(sync=True)
    y_min_pos = traitlets.Int(0).tag(sync=True)
    y_max_pos = traitlets.Int(100).tag(sync=True)
    z_min_pos = traitlets.Int(0).tag(sync=True)
    z_max_pos = traitlets.Int(100).tag(sync=True)

    x_thick_max = traitlets.Int(100).tag(sync=True)
    y_thick_max = traitlets.Int(100).tag(sync=True)
    z_thick_max = traitlets.Int(100).tag(sync=True)

    min_thickness = traitlets.Int(1).tag(sync=True)

    # Channels
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_dtypes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # Channel Parameters
    vmin_list = traitlets.List(traitlets.Any()).tag(sync=True)
    vmax_list = traitlets.List(traitlets.Any()).tag(sync=True)
    gamma_list = traitlets.List(traitlets.Float()).tag(sync=True)
    opacity_list = traitlets.List(traitlets.Float()).tag(sync=True)
    histograms_data = traitlets.List(traitlets.Dict()).tag(sync=True)

    # UI Toggles
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    sync_on_hover = traitlets.Bool(False).tag(sync=True)

    warning_msg = traitlets.Unicode("").tag(sync=True)

    # Communication
    hover_coords = traitlets.Dict().tag(sync=True) # {'plane': 'xy', 'x': 0.5, 'y': 0.5, 't': 123}
    axis_bounds = traitlets.Dict().tag(sync=True) # Bounding boxes of axes in figure coords

    # Save UI
    save_filename = traitlets.Unicode("filepath_save.svg").tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)

    def __init__(self, X, Y, Z, show_crosshair=True, sync_on_hover=False, **kwargs):
        super().__init__(**kwargs)
        self.show_crosshair = show_crosshair
        self.sync_on_hover = sync_on_hover
        self.dims = (Z, Y, X) # (Z, Y, X) convention from numpy shape

        # Set max thickness bounds
        self.x_thick_max = max(1, X - 1)
        self.y_thick_max = max(1, Y - 1)
        self.z_thick_max = max(1, Z - 1)

    def _init_observers(self):
        # Observe thickness changes to update position bounds
        self.observe(self._update_bounds, names=['x_t', 'y_t', 'z_t'])

        # Observe all parameters to update plot
        self.observe(self._render_wrapper, names=['x_s', 'y_s', 'z_s', 'x_t', 'y_t', 'z_t'])

        # Observe channel parameters
        self.observe(self._render_wrapper, names=['vmin_list', 'vmax_list', 'gamma_list', 'opacity_list'])

        # Observe crosshair toggle
        self.observe(self._render_wrapper, names=['show_crosshair'])

        # Observe hover synchronization
        self.observe(self._handle_hover_sync, names=['hover_coords'])

        # Observe save trigger
        self.observe(self._save_svg, names='save_trigger')

        # Initial bounds update
        self._update_bounds(None)

        # Initial render
        self._render_wrapper(None)

    def _update_bounds(self, change):
        # x
        lo = 0
        hi = max(0, self.dims[2] - 1)
        self.x_min_pos = lo
        self.x_max_pos = hi
        if self.x_s < lo: self.x_s = lo
        if self.x_s > hi: self.x_s = hi

        # y
        lo = 0
        hi = max(0, self.dims[1] - 1)
        self.y_min_pos = lo
        self.y_max_pos = hi
        if self.y_s < lo: self.y_s = lo
        if self.y_s > hi: self.y_s = hi

        # z
        lo = 0
        hi = max(0, self.dims[0] - 1)
        self.z_min_pos = lo
        self.z_max_pos = hi
        if self.z_s < lo: self.z_s = lo
        if self.z_s > hi: self.z_s = hi

    def _render_wrapper(self, change):
        fig = self._render()
        if fig:
            if len(fig.axes) >= 3:
                ax_xy = fig.axes[0]
                ax_zy = fig.axes[1]
                ax_xz = fig.axes[2]

                def get_bounds(ax):
                    bbox = ax.get_position()
                    return [bbox.x0, bbox.y0, bbox.width, bbox.height]

                self.axis_bounds = {
                    'xy': get_bounds(ax_xy),
                    'zy': get_bounds(ax_zy),
                    'xz': get_bounds(ax_xz)
                }

            buf = io.BytesIO()
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
            fig.savefig(buf, format='png')

            if len(fig.axes) >= 3:
                self.axis_bounds = {
                    'xy': get_bounds(fig.axes[0]),
                    'zy': get_bounds(fig.axes[1]),
                    'xz': get_bounds(fig.axes[2])
                }
            self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig) # Close to avoid memory leak

    def _handle_hover_sync(self, change):
        if not self.sync_on_hover:
            return

        coords = change.new if hasattr(change, 'new') else change.get('new', {})
        if not coords:
            return

        plane = coords.get('plane')
        frac_x = coords.get('x')
        frac_y = coords.get('y')

        bounds = self.axis_bounds.get(plane)
        if not bounds:
            return

        b_x0, b_y0, b_w, b_h = bounds

        # Check if click is inside this axis
        # Note: JS y_frac is from top-left. Matplotlib bounds are from bottom-left.
        mpl_y_frac = 1.0 - frac_y

        if not (b_x0 <= frac_x <= b_x0 + b_w and b_y0 <= mpl_y_frac <= b_y0 + b_h):
            return

        local_x = (frac_x - b_x0) / b_w
        local_y_mpl = (mpl_y_frac - b_y0) / b_h
        fraction_from_top = 1.0 - local_y_mpl

        if plane == 'xy':
            data_x = int(local_x * self.dims[2])
            data_y = int(fraction_from_top * self.dims[1])
            self.x_s = max(0, min(self.dims[2] - 1, data_x))
            self.y_s = max(0, min(self.dims[1] - 1, data_y))
        elif plane == 'zy':
            data_z = int(local_x * self.dims[0])
            data_y = int(fraction_from_top * self.dims[1])
            self.z_s = max(0, min(self.dims[0] - 1, data_z))
            self.y_s = max(0, min(self.dims[1] - 1, data_y))
        elif plane == 'xz':
            data_x = int(local_x * self.dims[2])
            data_z = int(fraction_from_top * self.dims[0])
            self.x_s = max(0, min(self.dims[2] - 1, data_x))
            self.z_s = max(0, min(self.dims[0] - 1, data_z))

    def _render(self):
        raise NotImplementedError

    def _save_svg(self, change):
        fig = self._render()
        if fig:
            try:
                fig.savefig(self.save_filename, format='svg', dpi=300, bbox_inches='tight')
                print(f"Saved to {self.save_filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
            finally:
                plt.close(fig)

class TNIASliceWidget(TNIAWidgetBase):
    def __init__(self, im, sxy=None, sz=None, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1, colors=None,
                 show_crosshair=True, sync_on_hover=False, x_s=None, y_s=None, z_s=None, x_t=None, y_t=None, z_t=None, opacity=None):
        if colors is not None:
            warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
            if colormap is None:
                colormap = colors

        # Handle 2D images gracefully by adding a Z dimension of 1
        if isinstance(im, list):
            if im[0].ndim == 2:
                im = [img[np.newaxis, ...] for img in im]
        elif im.ndim == 2:
            im = im[np.newaxis, ...]

        # Determine dimensions
        im_shape = (im[0].shape if isinstance(im, list) else im.shape)
        Z, Y, X = im_shape

        super().__init__(X, Y, Z, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover)

        self.im_orig = im

        self._sxy_given = sxy is not None
        self._sz_given = sz is not None
        self.sxy = sxy if self._sxy_given else 1.0
        self.sz = sz if self._sz_given else 1.0

        self.figsize = figsize
        self.colormap = colormap
        self.colors_orig = colors

        # Helper to ensure args are lists of length num_channels
        def _to_list(val, n, default):
            if val is None:
                return [default] * n
            elif isinstance(val, (list, tuple)):
                if len(val) >= n:
                    return list(val[:n])
                else:
                    return list(val) + [default] * (n - len(val))
            else:
                return [val] * n

        # Initialize Channel info
        if isinstance(im, list):
            self.num_channels = len(im)
            self.channel_names = [f"Channel {i}" for i in range(self.num_channels)]
            self.channel_dtypes = [img.dtype.name for img in im]

            # Resolve default colors to ensure stability when toggling
            if colormap is None:
                 defaults = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
                 # Extend if needed
                 while len(defaults) < self.num_channels:
                     defaults += defaults
                 self.colors_resolved = defaults[:self.num_channels]
            else:
                 self.colors_resolved = list(colormap)
        else:
            self.num_channels = 1
            self.channel_names = ["Channel 0"]
            self.channel_dtypes = [im.dtype.name]
            if colormap is None:
                self.colors_resolved = ['white']
            elif isinstance(colormap, (list, tuple)):
                self.colors_resolved = list(colormap)
            else:
                self.colors_resolved = [colormap]

        # Use resolve_color for channel_colors (which is passed to JS)
        self.channel_colors = [resolve_color(c) for c in self.colors_resolved]

        # Set traitlets lists for interactive parameters
        def _resolve_vmin(val, n):
            lst = _to_list(val, n, None)
            res = []
            for x in lst:
                res.append(0.0 if x is None else float(x))
            return res

        def _resolve_vmax(val, n, im_orig):
            lst = _to_list(val, n, None)
            res = []
            for i, x in enumerate(lst):
                if x is not None:
                    res.append(float(x))
                else:
                    if isinstance(im_orig, list):
                        m = float(np.nanmax(im_orig[i]))
                    else:
                        m = float(np.nanmax(im_orig))
                    if np.isnan(m):
                        m = 1.0
                    res.append(m)
            return res

        self.vmin_list = _resolve_vmin(vmin, self.num_channels)
        self.vmax_list = _resolve_vmax(vmax, self.num_channels, self.im_orig)
        self.gamma_list = _to_list(gamma, self.num_channels, 1.0)
        self.opacity_list = _to_list(opacity, self.num_channels, 1.0)

        # Set initial values if provided
        if x_t is not None: self.x_t = int(x_t)
        if y_t is not None: self.y_t = int(y_t)
        if z_t is not None: self.z_t = int(z_t)

        if x_s is not None: self.x_s = int(x_s)
        if y_s is not None: self.y_s = int(y_s)
        if z_s is not None: self.z_s = int(z_s)

        # Compute histograms
        hists = []
        if isinstance(im, list):
            for img in im:
                hists.append(compute_histogram(img))
        else:
            hists.append(compute_histogram(im))
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()

    def _render(self):
        Z, Y, X = self.dims

        x0 = max(0, self.x_s - self.x_t)
        x1 = min(X - 1, self.x_s + self.x_t)
        y0 = max(0, self.y_s - self.y_t)
        y1 = min(Y - 1, self.y_s + self.y_t)
        z0 = max(0, self.z_s - self.z_t)
        z1 = min(Z - 1, self.z_s + self.z_t)

        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        if z1 <= z0: z1 = z0 + 1

        x_lims = [x0, x1]
        y_lims = [y0, y1]
        z_lims = [z0, z1]

        clipped = False
        if x0 > self.x_s - self.x_t or x1 < self.x_s + self.x_t: clipped = True
        if y0 > self.y_s - self.y_t or y1 < self.y_s + self.y_t: clipped = True
        if z0 > self.z_s - self.z_t or z1 < self.z_s + self.z_t: clipped = True

        if clipped:
            self.warning_msg = "⚠️ Projection clipped to image boundaries"
        else:
            self.warning_msg = ""

        # Prepare arguments based on opacity
        # Channels with opacity == 0 are considered hidden and filtered out
        if isinstance(self.im_orig, list):
            # Map "" to None for vmin/vmax
            vmin_resolved = [None if v == "" else float(v) for v in self.vmin_list]
            vmax_resolved = [None if v == "" else float(v) for v in self.vmax_list]
            gamma_resolved = [float(g) for g in self.gamma_list]
            opacity_resolved = [float(o) for o in self.opacity_list]

            # Indices of visible channels
            visible_indices = [i for i, op in enumerate(opacity_resolved) if op > 0]

            if not visible_indices:
                # Create a placeholder figure
                fig = plt.figure(figsize=self.figsize if self.figsize else (10, 10))
                fig.text(0.5, 0.5, "No Channels Visible", ha='center', va='center', color='white')
                fig.patch.set_facecolor('black')
                return fig

            im_curr = [self.im_orig[i] for i in visible_indices]

            if isinstance(self.colors_resolved, list) and len(self.colors_resolved) == self.num_channels:
                 colors_curr = [self.colors_resolved[i] for i in visible_indices]
            else:
                 colors_curr = self.colors_resolved

            vmin_curr = [vmin_resolved[i] for i in visible_indices]
            vmax_curr = [vmax_resolved[i] for i in visible_indices]
            gamma_curr = [gamma_resolved[i] for i in visible_indices]
            opacity_curr = [opacity_resolved[i] for i in visible_indices]

        else:
            vmin_val = None if self.vmin_list[0] == "" else float(self.vmin_list[0])
            vmax_val = None if self.vmax_list[0] == "" else float(self.vmax_list[0])

            # Auto-calculate max if vmax is not explicitly set
            if vmax_val is None:
                vmax_val = float(np.max(self.im_orig))

            gamma_val = float(self.gamma_list[0])
            opacity_val = float(self.opacity_list[0])

            if opacity_val <= 0:
                fig = plt.figure(figsize=self.figsize if self.figsize else (10, 10))
                fig.text(0.5, 0.5, "No Channels Visible", ha='center', va='center', color='white')
                fig.patch.set_facecolor('black')
                return fig

            im_curr = self.im_orig
            colors_curr = self.colors_resolved
            vmin_curr = vmin_val
            vmax_curr = vmax_val
            gamma_curr = gamma_val
            opacity_curr = opacity_val

        # Pass None if not originally given to trigger scale bar logic
        pass_sxy = self.sxy if self._sxy_given else None
        pass_sz = self.sz if self._sz_given else None

        # show_zyx_max_slabs returns a Figure
        fig = show_zyx_max_slabs(
            im_curr, x_lims, y_lims, z_lims,
            sxy=pass_sxy, sz=pass_sz, figsize=self.figsize, colormap=colors_curr,
            vmin=vmin_curr, vmax=vmax_curr, gamma=gamma_curr, opacity=opacity_curr
        )

        # Crosshairs logic (copied from original interactive wrapper)
        if self.show_crosshair and fig:
            # XY
            fig.axes[0].axvline(x_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axvline(x_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            # ZY
            fig.axes[1].axvline(z_lims[0]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[0]*self.sxy + 0.5,     color='r', ls=':', alpha=0.3)
            fig.axes[1].axvline(z_lims[1]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[1]*self.sxy + 0.5,     color='r', ls=':', alpha=0.3)
            # XZ
            fig.axes[2].axvline(x_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[0]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[2].axvline(x_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[1]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)

        return fig


class TNIAAnnotatorWidget(TNIASliceWidget):
    """
    Subclass of TNIASliceWidget that supports interactive point annotation.
    """
    # UI Toggles
    annotation_mode = traitlets.Bool(False).tag(sync=True)
    annotation_action = traitlets.Unicode('add').tag(sync=True) # 'add' or 'delete'

    # Save CSV UI
    save_csv_filename = traitlets.Unicode("points.csv").tag(sync=True)
    save_csv_trigger = traitlets.Int(0).tag(sync=True)

    # Communication
    click_coords = traitlets.Dict().tag(sync=True) # {'plane': 'xy', 'x': 0.5, 'y': 0.5, 't': 123}

    # Data
    points = traitlets.List().tag(sync=True) # List of [z, y, x] lists

    def __init__(self, im, colormap=None, colors=None, opacity=None, point_size_scale=0.01, *args, **kwargs):
        if colors is not None:
            warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
            if colormap is None:
                colormap = colors
        # Normalize input to list
        if not isinstance(im, list):
            im_list = [im]
        else:
            im_list = list(im)

        # Ensure colormap is a list
        if colormap is None:
            if len(im_list) == 1:
                colors_list = ['white']
            else:
                colors_list = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
        elif isinstance(colormap, str):
            colors_list = [colormap]
        else:
            colors_list = list(colormap)

        while len(colors_list) < len(im_list):
            if len(im_list) == 1:
                colors_list.extend(['white'])
            else:
                colors_list.extend(['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue'])

        colors_list = colors_list[:len(im_list)]

        if opacity is None:
            opacity_list = [1.0] * len(im_list)
        elif isinstance(opacity, (int, float)):
            opacity_list = [opacity] * len(im_list)
        else:
            opacity_list = list(opacity)
            while len(opacity_list) < len(im_list):
                opacity_list.append(1.0)
            opacity_list = opacity_list[:len(im_list)]

        # Gracefully handle 2D
        if im_list[0].ndim == 2:
            im_list = [img[np.newaxis, ...] for img in im_list]

        # Get shape
        im_shape = im_list[0].shape
        Z, Y, X = im_shape
        min_dim = min(X, Y)
        self.point_size = max(3, int(np.ceil(point_size_scale * min_dim)))

        # Create persistent annotation channel
        self._annot_img = np.zeros((Z, Y, X), dtype=np.uint8)

        # Append annotation channel
        im_list.append(self._annot_img)
        colors_list.append('red')
        opacity_list.append(1.0)

        # Ensure vmax is padded correctly for the annotation channel
        vmax = kwargs.get('vmax', None)

        def resolve_vmax(img):
            m = float(np.nanmax(img))
            if np.isnan(m):
                m = 1.0
            return m

        if vmax is None:
            vmax_list = [resolve_vmax(im_list[i]) for i in range(len(im_list) - 1)] + [255.0]
            kwargs['vmax'] = vmax_list
        elif isinstance(vmax, (list, tuple)):
            vmax_list = list(vmax)
            while len(vmax_list) < len(im_list) - 1:
                vmax_list.append(None)
            vmax_list = vmax_list[:len(im_list) - 1]
            for i in range(len(vmax_list)):
                if vmax_list[i] is None:
                    vmax_list[i] = resolve_vmax(im_list[i])
            vmax_list.append(255.0)
            kwargs['vmax'] = vmax_list
        else:
            vmax_list = [vmax] * (len(im_list) - 1) + [255.0]
            kwargs['vmax'] = vmax_list

        # Initialize superclass
        super().__init__(im_list, colormap=colors_list, opacity=opacity_list, *args, **kwargs)

        # Override the last channel name
        self.channel_names = self.channel_names[:-1] + ['Annotations']

    def _init_observers(self):
        super()._init_observers()
        self.observe(self._handle_click, names=['click_coords'])
        self.observe(self._on_points_changed, names=['points'])
        # Also re-render if annotation_mode changes (so UI cursor updates)
        self.observe(self._render_wrapper, names=['annotation_mode'])
        self.observe(self._save_csv, names='save_csv_trigger')

    def _handle_click(self, change):
        if not self.annotation_mode:
            return

        coords = change.new if hasattr(change, 'new') else change.get('new', {})
        if not coords:
            return

        plane = coords.get('plane')
        frac_x = coords.get('x')
        frac_y = coords.get('y')

        bounds = self.axis_bounds.get(plane)
        if not bounds:
            return

        b_x0, b_y0, b_w, b_h = bounds

        # Check if click is inside this axis
        # Note: JS y_frac is from top-left. Matplotlib bounds are from bottom-left.
        mpl_y_frac = 1.0 - frac_y

        if not (b_x0 <= frac_x <= b_x0 + b_w and b_y0 <= mpl_y_frac <= b_y0 + b_h):
            return

        local_x = (frac_x - b_x0) / b_w
        local_y_mpl = (mpl_y_frac - b_y0) / b_h
        fraction_from_top = 1.0 - local_y_mpl

        x0 = max(0, self.x_s - self.x_t)
        x1 = min(self.dims[2] - 1, self.x_s + self.x_t)
        y0 = max(0, self.y_s - self.y_t)
        y1 = min(self.dims[1] - 1, self.y_s + self.y_t)
        z0 = max(0, self.z_s - self.z_t)
        z1 = min(self.dims[0] - 1, self.z_s + self.z_t)

        if plane == 'xy':
            data_x = int(local_x * self.dims[2])
            data_y = int(fraction_from_top * self.dims[1])
            data_z = self.z_s
        elif plane == 'zy':
            data_z = int(local_x * self.dims[0])
            data_y = int(fraction_from_top * self.dims[1])
            data_x = self.x_s
        elif plane == 'xz':
            data_x = int(local_x * self.dims[2])
            data_z = int(fraction_from_top * self.dims[0])
            data_y = self.y_s
        else:
            return

        data_x = max(0, min(self.dims[2] - 1, data_x))
        data_y = max(0, min(self.dims[1] - 1, data_y))
        data_z = max(0, min(self.dims[0] - 1, data_z))

        if self.annotation_action == 'add':
            self.add_point(data_z, data_y, data_x)
        elif self.annotation_action == 'delete':
            if not self.points: return

            pts = np.array(self.points)
            if plane == 'xy':
                mask = (pts[:, 0] >= z0) & (pts[:, 0] <= z1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 2] - data_x)**2 + (visible_pts[:, 1] - data_y)**2
            elif plane == 'zy':
                mask = (pts[:, 2] >= x0) & (pts[:, 2] <= x1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 0] - data_z)**2 + (visible_pts[:, 1] - data_y)**2
            elif plane == 'xz':
                mask = (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 2] - data_x)**2 + (visible_pts[:, 0] - data_z)**2

            closest_idx_in_visible = np.argmin(dist)
            closest_pt = visible_pts[closest_idx_in_visible]
            self.remove_point(closest_pt[0], closest_pt[1], closest_pt[2])

    def add_point(self, z, y, x):
        """Programmatically add a point"""
        pt = [int(z), int(y), int(x)]
        if pt not in self.points:
            self.points = self.points + [pt]

    def remove_point(self, z, y, x):
        """Programmatically remove a point"""
        pt = [int(z), int(y), int(x)]
        new_points = []
        deleted = False
        for p in self.points:
            if not deleted and p == pt:
                deleted = True
                continue
            new_points.append(p)
        if deleted:
            self.points = new_points

    def _save_csv(self, change):
        if not self.points:
            print("No points to save.")
            return

        import os
        filename = self.save_csv_filename
        if not filename:
            filename = "points.csv"

        filename = os.path.expanduser(os.path.expandvars(filename))

        try:
            with open(filename, 'w') as f:
                f.write("z,y,x\n")
                for p in self.points:
                    f.write(f"{p[0]},{p[1]},{p[2]}\n")
            print(f"Saved {len(self.points)} points to {filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")


    def _on_points_changed(self, change):
        # Update annotation mask efficiently
        self._annot_img.fill(0)
        Z, Y, X = self.dims
        s = max(1, self.point_size // 2)
        R = s * min(self.sxy, self.sz)
        s_xy = max(1, int(np.round(R / self.sxy)))
        s_z = max(1, int(np.round(R / self.sz)))
        for p in self.points:
            pz, py, px = p
            z0 = max(0, pz - s_z)
            z1 = min(Z, pz + s_z + 1)
            y0 = max(0, py - s_xy)
            y1 = min(Y, py + s_xy + 1)
            x0 = max(0, px - s_xy)
            x1 = min(X, px + s_xy + 1)
            self._annot_img[z0:z1, y0:y1, x0:x1] = 255

        self._render_wrapper(change)

class TNIAScatterWidget(TNIAWidgetBase):
    def __init__(self, X_arr, Y_arr, Z_arr, channels=None, sxy=None, sz=None, render='points', bins=512,
                 point_size=4, alpha=0.6, colormap=None, colors=None, opacity=None, gamma=1, vmin=None, vmax=None, figsize=None,
                 show_crosshair=True, sync_on_hover=False, subplot_bg='black', x_s=None, y_s=None, z_s=None, x_t=None, y_t=None, z_t=None):
        if colors is not None:
            warnings.warn("The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead.", DeprecationWarning, stacklevel=2)
            if colormap is None:
                colormap = colors

        self.X_arr = np.asarray(X_arr)
        self.Y_arr = np.asarray(Y_arr)
        self.Z_arr = np.asarray(Z_arr)

        # Compute bounds for init
        xmin, xmax = float(np.floor(self.X_arr.min())), float(np.ceil(self.X_arr.max()))
        ymin, ymax = float(np.floor(self.Y_arr.min())), float(np.ceil(self.Y_arr.max()))
        zmin, zmax = float(np.floor(self.Z_arr.min())), float(np.ceil(self.Z_arr.max()))

        X_dim = int(np.ceil(xmax - xmin + 1))
        Y_dim = int(np.ceil(ymax - ymin + 1))
        Z_dim = int(np.ceil(zmax - zmin + 1))

        super().__init__(X_dim, Y_dim, Z_dim, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover)

        self.channels = channels
        self._sxy_given = sxy is not None
        self._sz_given = sz is not None
        self.sxy = sxy if self._sxy_given else 1.0
        self.sz = sz if self._sz_given else 1.0
        self.render = render
        self.bins = bins
        self.point_size = point_size
        self.alpha = alpha
        self.colors = colors
        self.figsize = figsize

        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.subplot_bg = subplot_bg

        # Precompute Density helpers
        def _resolve_bins(B):
            if isinstance(B, (tuple, list)) and len(B) == 2:
                bx, by = int(B[0]), int(B[1])
            else:
                bx = by = int(B)
            bx = max(1, bx); by = max(1, by)
            return bx, by

        self.BX, self.BY = _resolve_bins(bins)

        # Analyze channels
        self.N = len(self.X_arr)

        def _is_int_like(a): return np.issubdtype(np.asarray(a).dtype, np.integer) or np.asarray(a).dtype == bool
        def _as_1d(a):
            a = np.asarray(a)
            return a.reshape(-1) if a.ndim == 1 else a

        self.mode = 'single'
        self.ch_ids = None
        self.cont_single = None
        self.cont_multi = None
        self.idx_lists = None

        if channels is None:
            self.mode = 'single'
        elif isinstance(channels, (list, tuple)):
            arrs = [ _as_1d(a) for a in channels ]
            lens = {len(a) for a in arrs}
            if lens == {self.N} and any(np.issubdtype(a.dtype, np.floating) for a in arrs):
                self.mode = 'cont_multi'
                self.cont_multi = np.stack([a.astype(float) for a in arrs], axis=1)
            else:
                self.mode = 'idx_lists'
                self.idx_lists = [np.asarray(ix, dtype=int) for ix in channels]
        else:
            a = _as_1d(channels)
            if _is_int_like(a):
                self.mode = 'ids'
                self.ch_ids = np.asarray(a, dtype=int)
            else:
                self.mode = 'cont_single'
                self.cont_single = np.asarray(a, dtype=float)

        if self.mode in ('single',):
            self.C = 1
        elif self.mode == 'ids':
            self.C = int(self.ch_ids.max()) + 1 if self.ch_ids.size else 1
        elif self.mode == 'idx_lists':
            self.C = len(self.idx_lists)
        elif self.mode == 'cont_single':
            self.C = 1
        else:
            self.C = self.cont_multi.shape[1]

        if colormap is None:
            if self.C == 1:
                default_cols = ['white']
            else:
                default_cols = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
            while len(default_cols) < self.C:
                 default_cols += default_cols
            self.colors_use = default_cols[:max(1, self.C)]
        else:
            if isinstance(colormap, str):
                self.colors_use = [colormap]
            elif isinstance(colormap, (list, tuple)):
                self.colors_use = list(colormap)
            else:
                self.colors_use = [colormap]

        if self.mode == 'ids' and len(self.colors_use) == 1 and is_colormap(self.colors_use[0]):
            cmap = plt.get_cmap(self.colors_use[0])
            self.colors_rgb = [cmap(i / max(1, self.C - 1))[:3] for i in range(self.C)]
        else:
            self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]


        # Set traitlets lists for interactive parameters
        def _to_list(val, n, default):
            if val is None:
                return [default] * n
            elif isinstance(val, (list, tuple)):
                if len(val) >= n:
                    return list(val[:n])
                else:
                    return list(val) + [default] * (n - len(val))
            else:
                return [val] * n

        def _resolve_vmin(val, n):
            lst = _to_list(val, n, None)
            res = []
            for x in lst:
                res.append(0.0 if x is None else float(x))
            return res

        def _resolve_vmax(val, n, mode, cont_single, cont_multi):
            lst = _to_list(val, n, None)
            res = []
            for i, x in enumerate(lst):
                if x is not None:
                    res.append(float(x))
                else:
                    if mode == 'cont_single':
                        m = float(np.nanmax(cont_single))
                    elif mode == 'cont_multi':
                        m = float(np.nanmax(cont_multi[:, i]))
                    else:
                        m = 1.0

                    if np.isnan(m):
                        m = 1.0
                    res.append(m)
            return res

        self.vmin_list = _resolve_vmin(vmin, self.C)
        self.vmax_list = _resolve_vmax(vmax, self.C, self.mode, self.cont_single, self.cont_multi)
        self.gamma_list = _to_list(gamma, self.C, 1.0)
        self.opacity_list = _to_list(opacity, self.C, 1.0)

        self.channel_names = [f"Channel {i}" for i in range(self.C)]
        self.channel_dtypes = ["float"] * self.C
        self.channel_colors = [matplotlib.colors.to_hex(c) for c in self.colors_rgb]

        # Init values
        if x_t is not None: self.x_t = int(x_t)
        if y_t is not None: self.y_t = int(y_t)
        if z_t is not None: self.z_t = int(z_t)

        # Center default
        if x_s is None: self.x_s = int((self.xmax - self.xmin)/2) # relative to 0
        else: self.x_s = int(x_s)

        if y_s is None: self.y_s = int((self.ymax - self.ymin)/2)
        else: self.y_s = int(y_s)

        if z_s is None: self.z_s = int((self.zmax - self.zmin)/2)
        else: self.z_s = int(z_s)

        # Compute histograms
        hists = []
        if self.mode == 'single' or self.mode == 'ids' or self.mode == 'idx_lists':
            # For scatter with a single set of points and uniform color, the distribution is just counts (not really intensities).
            # We don't have multiple intensity channels here. We'll leave it empty.
            # If ids, we just have 1 channel (the IDs).
            if self.mode == 'ids':
                hists.append(compute_histogram(self.ch_ids))
            else:
                hists.append({'counts': [], 'bin_edges': []}) # No intensities to map
        elif self.mode == 'cont_single':
            hists.append(compute_histogram(self.cont_single))
        elif self.mode == 'cont_multi':
            for c in range(self.C):
                hists.append(compute_histogram(self.cont_multi[:, c]))

        # pad to C
        while len(hists) < self.C:
            hists.append({'counts': [], 'bin_edges': []})
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()

    def _render(self):
        vmin_resolved = [None if v == "" else float(v) for v in self.vmin_list]
        vmax_resolved = [None if v == "" else float(v) for v in self.vmax_list]
        gamma_resolved = [float(g) for g in self.gamma_list]
        opacity_resolved = [float(o) for o in self.opacity_list]

        # Translate widget relative coordinates (0..Dim) to data coordinates (min..max)
        x_c = self.x_s + self.xmin
        y_c = self.y_s + self.ymin
        z_c = self.z_s + self.zmin

        x0 = max(self.xmin, x_c - self.x_t)
        x1 = min(self.xmax, x_c + self.x_t)
        y0 = max(self.ymin, y_c - self.y_t)
        y1 = min(self.ymax, y_c + self.y_t)
        z0 = max(self.zmin, z_c - self.z_t)
        z1 = min(self.zmax, z_c + self.z_t)

        x_lims = (x0, x1)
        y_lims = (y0, y1)
        z_lims = (z0, z1)

        clipped = False
        if x0 > x_c - self.x_t or x1 < x_c + self.x_t: clipped = True
        if y0 > y_c - self.y_t or y1 < y_c + self.y_t: clipped = True
        if z0 > z_c - self.z_t or z1 < z_c + self.z_t: clipped = True

        if clipped:
            self.warning_msg = "⚠️ Projection clipped to data boundaries"
        else:
            self.warning_msg = ""

        # Density Mode Logic
        if self.render == 'density':
            EMPTY_XY = np.zeros((self.BY, self.BX), dtype=float)
            EMPTY_XZ = np.zeros((self.BY, self.BX), dtype=float)
            EMPTY_ZY = np.zeros((self.BY, self.BX), dtype=float)

            def _hist2d(x, y, xr, yr, w=None):
                H, _, _ = np.histogram2d(y, x, bins=[self.BY, self.BX], range=[yr, xr], weights=w)
                return H

            xy_list, xz_list, zy_list = [], [], []

            if self.mode in ('single', 'ids', 'idx_lists'):
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for idxs in idx_lists_local:
                    if idxs.size == 0:
                        xy_list.append(EMPTY_XY.copy()); xz_list.append(EMPTY_XZ.copy()); zy_list.append(EMPTY_ZY.copy()); continue
                    Xi, Yi, Zi = self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    Hxy = _hist2d(Xi[mZ], Yi[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1)) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(Xi[mY], Zi[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1)) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(Zi[mX], Yi[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1)) if mX.any() else EMPTY_ZY.copy()
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            elif self.mode == 'cont_single':
                vals = self.cont_single
                mZ = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
                mY = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
                mX = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

                Hxy = _hist2d(self.X_arr[mZ], self.Y_arr[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1), w=vals[mZ]) if mZ.any() else EMPTY_XY.copy()
                Hxz = _hist2d(self.X_arr[mY], self.Z_arr[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1), w=vals[mY]) if mY.any() else EMPTY_XZ.copy()
                Hzy = _hist2d(self.Z_arr[mX], self.Y_arr[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1), w=vals[mX]) if mX.any() else EMPTY_ZY.copy()

                xy_list = [Hxy]; xz_list = [Hxz]; zy_list = [Hzy]

            else: # cont_multi
                vals = self.cont_multi
                for c in range(self.C):
                    v = vals[:, c]
                    mZ = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
                    mY = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
                    mX = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

                    Hxy = _hist2d(self.X_arr[mZ], self.Y_arr[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1), w=v[mZ]) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(self.X_arr[mY], self.Z_arr[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1), w=v[mY]) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(self.Z_arr[mX], self.Y_arr[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1), w=v[mX]) if mX.any() else EMPTY_ZY.copy()
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            if self.mode in ('single', 'ids', 'idx_lists', 'cont_multi'):
                colors_for_rgb = self.colors_use
            else:
                colors_for_rgb = self.colors_use

            has_colormap = any(is_colormap(c) for c in colors_for_rgb)
            if has_colormap:
                xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb_cmap(
                    xy_list, xz_list, zy_list,
                    vmin=vmin_resolved, vmax=vmax_resolved, gamma=gamma_resolved, colormap=colors_for_rgb, opacity=opacity_resolved, blend='add', soft_clip=True
                )
            else:
                xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
                    xy_list, xz_list, zy_list,
                    vmin=vmin_resolved, vmax=vmax_resolved, gamma=gamma_resolved, colormap=colors_for_rgb, opacity=opacity_resolved, blend='add', soft_clip=True
                )

            pass_sxy = self.sxy if self._sxy_given else None
            pass_sz = self.sz if self._sz_given else None

            fig = show_zyx(
                xy_rgb, xz_rgb, zy_rgb,
                sxy=pass_sxy, sz=pass_sz, figsize=self.figsize, colormap=None,
                vmin=None, vmax=None, gamma=1, use_plt=True, colors=None, opacity=opacity_resolved, subplot_bg=self.subplot_bg
            )

            fig.patch.set_alpha(0.0) # transparent figure bg
            fig.patch.set_facecolor("none")
            # Subplot facecolor is handled by show_zyx if subplot_bg is passed

            # Crosshairs
            if self.show_crosshair:
                axXY, axZY, axXZ = fig.axes[0], fig.axes[1], fig.axes[2]
                axXY.vlines([x_lims[0]*self.sxy + 0.5, (x_lims[1]+1)*self.sxy + 0.5], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axXY.hlines([y_lims[0]*self.sxy + 0.5, (y_lims[1]+1)*self.sxy + 0.5], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.vlines([z_lims[0]*self.sz + 0.5*self.sz, (z_lims[1]+1)*self.sz + 0.5*self.sz], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.hlines([y_lims[0]*self.sxy + 0.5,       (y_lims[1]+1)*self.sxy + 0.5], self.zmin*self.sz, (self.zmax+1)*self.sz, colors='r', linestyles=':', alpha=0.3)
                axXZ.vlines([x_lims[0]*self.sxy + 0.5, (x_lims[1]+1)*self.sxy + 0.5], self.zmin*self.sz, (self.zmax+1)*self.sz, colors='r', linestyles=':', alpha=0.3)
                axXZ.hlines([z_lims[0]*self.sz + 0.5*self.sz, (z_lims[1]+1)*self.sz + 0.5*self.sz], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)

            return fig

        else:
            # Points mode
            z_xy_ratio = (self.sz / self.sxy) if self.sxy != self.sz else 1
            width_ratios  = [int(self.xmax - self.xmin + 1), int((self.zmax - self.zmin + 1) * z_xy_ratio)]
            height_ratios = [int(self.ymax - self.ymin + 1), int((self.zmax - self.zmin + 1) * z_xy_ratio)]

            fig, axs = plt.subplots(
                2, 2, figsize=self.figsize, constrained_layout=False,
                gridspec_kw=dict(width_ratios=width_ratios, height_ratios=height_ratios),
                facecolor='none'
            )
            axXY, axZY = axs[0,0], axs[0,1]
            axXZ, axBar = axs[1,0], axs[1,1]
            for ax in (axXY, axZY, axXZ, axBar):
                if ax is not axBar and self.subplot_bg is not None:
                    ax.set_facecolor(self.subplot_bg)
                else:
                    ax.patch.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            mZ_all = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
            mY_all = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
            mX_all = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

            if self.mode == 'cont_single':
                vals = self.cont_single
                c_use = self.colors_use[0]
                if is_colormap(c_use):
                    cmap = plt.get_cmap(c_use)
                else:
                    cmap = black_to(resolve_color(c_use))

                vmin_c = np.nanmin(vals) if vmin_resolved[0] is None else vmin_resolved[0]
                vmax_c = np.nanmax(vals) if vmax_resolved[0] is None else vmax_resolved[0]
                norm = matplotlib.colors.Normalize(vmin=vmin_c, vmax=vmax_c)

                if mZ_all.any():
                    axXY.scatter(self.X_arr[mZ_all]*self.sxy, self.Y_arr[mZ_all]*self.sxy, c=vals[mZ_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(self.X_arr[mY_all]*self.sxy, self.Z_arr[mY_all]*self.sz,  c=vals[mY_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(self.Z_arr[mX_all]*self.sz,  self.Y_arr[mX_all]*self.sxy, c=vals[mX_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)

            elif self.mode == 'cont_multi':
                cols = blend_colors(self.cont_multi, self.colors_use, vmin=vmin_resolved, vmax=vmax_resolved, gamma=gamma_resolved, soft_clip=True)
                if mZ_all.any():
                    axXY.scatter(self.X_arr[mZ_all]*self.sxy, self.Y_arr[mZ_all]*self.sxy, c=cols[mZ_all], s=self.point_size, alpha=self.alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(self.X_arr[mY_all]*self.sxy, self.Z_arr[mY_all]*self.sz,  c=cols[mY_all], s=self.point_size, alpha=self.alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(self.Z_arr[mX_all]*self.sz,  self.Y_arr[mX_all]*self.sxy, c=cols[mX_all], s=self.point_size, alpha=self.alpha, linewidths=0)

            else:
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for c, idxs in enumerate(idx_lists_local):
                    if idxs.size == 0: continue
                    Xi, Yi, Zi = self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs]
                    col = [self.colors_rgb[c % len(self.colors_rgb)]]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    if mZ.any(): axXY.scatter(Xi[mZ]*self.sxy, Yi[mZ]*self.sxy, s=self.point_size, c=col, alpha=self.alpha, linewidths=0)
                    if mY.any(): axXZ.scatter(Xi[mY]*self.sxy, Zi[mY]*self.sz,  s=self.point_size, c=col, alpha=self.alpha, linewidths=0)
                    if mX.any(): axZY.scatter(Zi[mX]*self.sz,  Yi[mX]*self.sxy, s=self.point_size, c=col, alpha=self.alpha, linewidths=0)

            # Axis limits
            axXY.set_xlim([self.xmin*self.sxy, (self.xmax+1)*self.sxy]); axXY.set_ylim([(self.ymax+1)*self.sxy, self.ymin*self.sxy])
            axXZ.set_xlim([self.xmin*self.sxy, (self.xmax+1)*self.sxy]); axXZ.set_ylim([(self.zmax+1)*self.sz,  self.zmin*self.sz ])
            axZY.set_xlim([self.zmin*self.sz,  (self.zmax+1)*self.sz ]); axZY.set_ylim([(self.ymax+1)*self.sxy, self.ymin*self.sxy])

            if self.show_crosshair:
                axXY.vlines([x_lims[0]*self.sxy, (x_lims[1]+1)*self.sxy], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axXY.hlines([y_lims[0]*self.sxy, (y_lims[1]+1)*self.sxy], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.vlines([z_lims[0]*self.sz,  (z_lims[1]+1)*self.sz],  self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.hlines([y_lims[0]*self.sxy, (y_lims[1]+1)*self.sxy], self.zmin*self.sz,   (self.zmax+1)*self.sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.vlines([x_lims[0]*self.sxy, (x_lims[1]+1)*self.sxy], self.zmin*self.sz,   (self.zmax+1)*self.sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.hlines([z_lims[0]*self.sz,  (z_lims[1]+1)*self.sz],  self.xmin*self.sxy,  (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)

            # Scale bar (kept opaque)
            fig.patch.set_alpha(1.0)
            width_um = (self.xmax - self.xmin + 1) * self.sxy
            target = width_um * 0.2
            def nice_length(x):
                exp = np.floor(np.log10(x))
                for m in [5,2,1]:
                    val = m * 10**exp
                    if val <= x: return val
                return x
            bar_um = nice_length(target)
            bar_pix = bar_um / self.sxy
            bar_frac = bar_pix / (self.xmax - self.xmin + 1)
            fig_h_in = self.figsize[1] if self.figsize else 10
            fontsize_pt = max(8, min(24, fig_h_in * 72 * 0.03))
            x0 = 0.5 - bar_frac/2; x1 = 0.5 + bar_frac/2; y = 0.5
            axBar.hlines(y, x0, x1, transform=axBar.transAxes, linewidth=2, color='gray')

            both_given = self._sxy_given and self._sz_given
            if both_given:
                text_label = f"{int(bar_um)} µm"
            else:
                text_label = "`sxy` , `sz`"

            axBar.text(0.5, y - 0.1, text_label, transform=axBar.transAxes,
                       ha='center', va='top', color='gray', fontsize=fontsize_pt)

            fig.tight_layout(pad=0.0)
            return fig


def show_zyx_max_slice_interactive(
    im,
    sxy=None, sz=None,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True, sync_on_hover=False,
    colors=None, opacity=None,
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    """
    Interactive 3D slice viewer using AnyWidget.

    Inspired by show_zyx_max_slice_interactive in tnia_plotting_3d.py (ipywidgets version).

    Notes:
        - When `vmax` is not explicitly provided, it defaults to the 99.9th percentile for float arrays,
          and `np.max` for all integer and boolean arrays to prevent clipping. `vmin` defaults to 0 or
          its current minimum behavior.
        - Interactive widgets compute a 128-bin histogram per channel in the Python backend (excluding NaNs for floats)
          and synchronize it with the JS frontend via a `histograms_data` traitlet. The frontend renders this on a `<canvas>`
          underneath the channel controls, overlaid with a curve reflecting the current `vmin`, `vmax`, and `gamma` settings.
    """
    if isinstance(im, list):
        if im[0].ndim == 2:
            im = [img[np.newaxis, ...] for img in im]
    elif im.ndim == 2:
        im = im[np.newaxis, ...]
    im_shape = (im[0].shape if isinstance(im, list) else im.shape)
    Z, Y, X = im_shape
    _sxy = sxy if sxy is not None else 1
    _sz = sz if sz is not None else 1
    z_xy_ratio = (_sz / _sxy) if _sxy != _sz else 1

    if figsize is None:
        width_px  = X + Z * z_xy_ratio
        height_px = Y + Z * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, n // 64)
    if x_t is None: x_t = _default_t(X)
    if y_t is None: y_t = _default_t(Y)
    if z_t is None: z_t = _default_t(Z)

    # Defaults for s are handled in init (midpoint)
    if x_s is None: x_s = X // 2
    if y_s is None: y_s = Y // 2
    if z_s is None: z_s = Z // 2

    # Override for max projection if thickness exceeds shape
    if x_t >= X:
        x_t = max(1, X // 2)
        x_s = X // 2
    if y_t >= Y:
        y_t = max(1, Y // 2)
        y_s = Y // 2
    if z_t >= Z:
        z_t = max(1, Z // 2)
        z_s = Z // 2

    return TNIASliceWidget(
        im, sxy=sxy, sz=sz, figsize=figsize, colormap=colormap,
        vmin=vmin, vmax=vmax, gamma=gamma, colors=colors, opacity=opacity, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover,
        x_s=x_s, y_s=y_s, z_s=z_s, x_t=x_t, y_t=y_t, z_t=z_t
    )

def show_zyx_max_slice_interactive_point_annotator(
    im,
    sxy=None, sz=None,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True,
    colors=None, opacity=None,
    point_size_scale=0.01,
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    """
    Interactive 3D slice viewer with point annotation using AnyWidget.
    Features the same visualization as show_zyx_max_slice_interactive, but adds:
    - Point annotation toggle
    - Left-click on any projection to add or delete points.
    Returns a widget instance `w`. The list of annotated points is accessible via `w.points`.
    """
    if isinstance(im, list):
        if im[0].ndim == 2:
            im = [img[np.newaxis, ...] for img in im]
    elif im.ndim == 2:
        im = im[np.newaxis, ...]
    im_shape = (im[0].shape if isinstance(im, list) else im.shape)
    Z, Y, X = im_shape
    _sxy = sxy if sxy is not None else 1
    _sz = sz if sz is not None else 1
    z_xy_ratio = (_sz / _sxy) if _sxy != _sz else 1

    if figsize is None:
        width_px  = X + Z * z_xy_ratio
        height_px = Y + Z * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, n // 64)
    if x_t is None: x_t = _default_t(X)
    if y_t is None: y_t = _default_t(Y)
    if z_t is None: z_t = _default_t(Z)

    if x_s is None: x_s = X // 2
    if y_s is None: y_s = Y // 2
    if z_s is None: z_s = Z // 2

    if x_t >= X:
        x_t = max(1, X // 2)
        x_s = X // 2
    if y_t >= Y:
        y_t = max(1, Y // 2)
        y_s = Y // 2
    if z_t >= Z:
        z_t = max(1, Z // 2)
        z_s = Z // 2

    return TNIAAnnotatorWidget(
        im, sxy=sxy, sz=sz, figsize=figsize, colormap=colormap,
        vmin=vmin, vmax=vmax, gamma=gamma, colors=colors, opacity=opacity, show_crosshair=show_crosshair,
        point_size_scale=point_size_scale,
        x_s=x_s, y_s=y_s, z_s=z_s, x_t=x_t, y_t=y_t, z_t=z_t
    )

def show_zyx_max_scatter_interactive(
    points,
    channels=None,
    sxy=None, sz=None,
    render=None,
    bins=512,
    point_size=4, alpha=0.6,
    colormap=None, colors=None, opacity=None,
    gamma=1, vmin=None, vmax=None,
    figsize=None, figsize_scale=1.0,
    show_crosshair=True, sync_on_hover=False,
    subplot_bg='black',
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    """
    Shows interactive sliders for XY, XZ, and YZ projection of 3D point coordinates.

    Notes:
        - When hiding axes to maintain custom subplot backgrounds (e.g. `subplot_bg`), the code
          manually hides ticks and spines instead of using `ax.axis('off')`, as `ax.axis('off')`
          inadvertently disables the background patch in matplotlib.
        - When `vmax` is not explicitly provided, it defaults to the 99.9th percentile for float arrays,
          and `np.max` for all integer and boolean arrays to prevent clipping. `vmin` defaults to 0 or
          its current minimum behavior.
        - The function dynamically determines its rendering mode: if `render=None`, it defaults to
          `'points'` for point counts under 10,000, and `'density'` for larger datasets to optimize interactive performance.
    """
    if isinstance(points, (tuple, list)) and len(points) == 3:
        # points is a tuple or list of 3 arrays: (Z, Y, X)
        Z, Y, X = points[0], points[1], points[2]
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)
    else:
        points = np.asarray(points)
        if points.ndim == 2 and points.shape[1] == 3:
            # points shape is (N, 3), assume Z, Y, X
            Z, Y, X = points[:, 0], points[:, 1], points[:, 2]
        elif points.ndim == 2 and points.shape[0] == 3:
            # points shape is (3, N), assume Z, Y, X
            Z, Y, X = points[0], points[1], points[2]
        elif points.ndim == 1 and len(points) == 3:
            # array of 3 list-like objects
            Z, Y, X = points[0], points[1], points[2]
            X = np.asarray(X)
            Y = np.asarray(Y)
            Z = np.asarray(Z)
        else:
            raise ValueError("points must be an array of shape (N, 3) representing (Z, Y, X) or a tuple/list of 3 arrays (Z, Y, X).")

    if render is None:
        if len(X) < 10000:
            render = 'points'
        else:
            render = 'density'


    xmin, xmax = float(np.floor(X.min())), float(np.ceil(X.max()))
    ymin, ymax = float(np.floor(Y.min())), float(np.ceil(Y.max()))
    zmin, zmax = float(np.floor(Z.min())), float(np.ceil(Z.max()))

    XN = xmax - xmin + 1
    YN = ymax - ymin + 1
    ZN = zmax - zmin + 1
    _sxy = sxy if sxy is not None else 1
    _sz = sz if sz is not None else 1
    z_xy_ratio = (_sz / _sxy) if _sxy != _sz else 1

    if figsize is None:
        width_px  = XN + ZN * z_xy_ratio
        height_px = YN + ZN * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, int(n // 64))
    Xdim = int(np.ceil(XN)); Ydim = int(np.ceil(YN)); Zdim = int(np.ceil(ZN))

    if x_t is None: x_t = _default_t(Xdim)
    if y_t is None: y_t = _default_t(Ydim)
    if z_t is None: z_t = _default_t(Zdim)

    # Note: s inputs to scatter are in data coords (min-max range)
    # The Widget expects 0-Dim range?
    # Wait, my logic in TNIAScatterWidget._render adds xmin to x_s.
    # So x_s inside the widget is 0-based offset.
    # But x_s PASSED here is likely data coord?
    # Original code:
    # x_center_default = int(np.round((xmin + xmax) * 0.5))
    # x_s0 = _clip(int(x_s if x_s is not None else x_center_default), x_lo0, x_hi0)
    # So original input x_s is data coordinate.

    # So if x_s is provided, I need to subtract xmin to get 0-based offset for the widget init.
    if x_s is not None: x_s = int(x_s - xmin)
    if y_s is not None: y_s = int(y_s - ymin)
    if z_s is not None: z_s = int(z_s - zmin)

    return TNIAScatterWidget(
        X, Y, Z,
        channels=channels, sxy=sxy, sz=sz, render=render, bins=bins,
        point_size=point_size, alpha=alpha, colors=colors, opacity=opacity,
        gamma=gamma, vmin=vmin, vmax=vmax, figsize=figsize, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover,
        subplot_bg=subplot_bg,
        x_s=x_s, y_s=y_s, z_s=z_s, x_t=x_t, y_t=y_t, z_t=z_t
    )

class IsoScatterWidget(anywidget.AnyWidget):
    _esm = ir.files("eigenp_utils").joinpath("iso_scatter.js")

    image_data = traitlets.Unicode().tag(sync=True)
    elev = traitlets.Float(30).tag(sync=True)
    azim = traitlets.Float(-60).tag(sync=True)
    save_filename = traitlets.Unicode("iso_scatter.svg").tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)

    def __init__(self, X, Y, Z, color=None, sxy=1, sz=1, figsize=(12, 10),
                 point_size=5, alpha=0.6, cmap='viridis', max_points=10000,
                 title=None):
        super().__init__()

        # Store ORIGINAL Data
        self.X_orig = np.asarray(X)
        self.Y_orig = np.asarray(Y)
        self.Z_orig = np.asarray(Z)

        # Determine color mode and convert to numpy array immediately
        self.is_continuous = False
        self.is_categorical = False

        # Ensure color is numpy array
        if color is not None:
             self.color = np.asarray(color)
        else:
             self.color = None

        # Handle empty data
        if len(self.X_orig) == 0:
             self.cx, self.cy, self.cz = 0, 0, 0
             self.max_radius = 1.0
             self.sxy = sxy
             self.sz = sz
             self.figsize = figsize
             self.point_size = point_size
             self.alpha = alpha
             self.cmap = cmap
             self.title = title
             # Initialize rendering observers even if empty
             self.observe(self._render_wrapper, names=['elev', 'azim'])
             self.observe(self._save_svg, names='save_trigger')
             self._render_wrapper(None)
             return

        # Subsample if needed
        if len(self.X_orig) > max_points:
            idx = np.random.choice(len(self.X_orig), max_points, replace=False)
            self.X_orig = self.X_orig[idx]
            self.Y_orig = self.Y_orig[idx]
            self.Z_orig = self.Z_orig[idx]
            if self.color is not None:
                 self.color = self.color[idx]

        self.sxy = sxy
        self.sz = sz
        self.figsize = figsize
        self.point_size = point_size
        self.alpha = alpha
        self.cmap = cmap
        self.title = title

        # Calculate Centroid and Fixed Bounds for Rotation
        self.cx = (self.X_orig.min() + self.X_orig.max()) / 2
        self.cy = (self.Y_orig.min() + self.Y_orig.max()) / 2
        self.cz = (self.Z_orig.min() + self.Z_orig.max()) / 2

        # Calculate max radius from centroid
        dx = (self.X_orig - self.cx) * sxy
        dy = (self.Y_orig - self.cy) * sxy
        dz = (self.Z_orig - self.cz) * sz
        dists = np.sqrt(dx**2 + dy**2 + dz**2)
        self.max_radius = dists.max() * 1.05 # 5% buffer

        # Analyze color types (after subsampling)
        if self.color is not None:
            if np.issubdtype(self.color.dtype, np.number):
                 self.is_continuous = True
            else:
                 self.is_categorical = True
                 # Map categories to colors
                 self.unique_cats = np.unique(self.color)
                 n_cats = len(self.unique_cats)

                 # Use a distinct colormap for categorical
                 # We'll generate a color mapping dictionary
                 # Use matplotlib 'tab10' or 'tab20' or 'viridis' if too many
                 base_cmap = plt.get_cmap('tab20' if n_cats > 10 else 'tab10')
                 self.cat_color_map = {cat: base_cmap(i % base_cmap.N) for i, cat in enumerate(self.unique_cats)}
                 self.c_mapped = [self.cat_color_map[c] for c in self.color]

        self.observe(self._render_wrapper, names=['elev', 'azim'])
        self.observe(self._save_svg, names='save_trigger')

        # Initial render
        self._render_wrapper(None)

    def _render_wrapper(self, change):
        fig = self._render()
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)

    def _save_svg(self, change):
        fig = self._render()
        if fig:
            try:
                fig.savefig(self.save_filename, format='svg', dpi=300, bbox_inches='tight')
                print(f"Saved to {self.save_filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
            finally:
                plt.close(fig)

    def _get_rotation_matrix(self, elev_deg, azim_deg):
        # Convert to radians
        elev = np.radians(elev_deg)
        azim = np.radians(azim_deg)

        # Rotation around Z (Azimuth)
        Rz = np.array([
            [np.cos(azim), -np.sin(azim), 0],
            [np.sin(azim),  np.cos(azim), 0],
            [0,             0,            1]
        ])

        # Rotation around X (Elevation) - or Y?
        # Usually Elevation is tilting up/down.
        # If Z is up, X is right, Y is forward (or similar).
        # Rotation around X tilts Y and Z.
        Rx = np.array([
            [1, 0,             0],
            [0, np.cos(elev), -np.sin(elev)],
            [0, np.sin(elev),  np.cos(elev)]
        ])

        # Combined Rotation
        return Rx @ Rz

    def _render(self):
        fig = plt.figure(figsize=self.figsize, facecolor='white')

        # Layout: 3 Rows, 2 Columns
        # Row 0 (Span 2): 3D Scatter
        # Row 1 (0): Side View (Y-Z)
        # Row 1 (1): Front View (X-Z)
        # Row 2 (0): Top View (X-Y)
        # Row 2 (1): Legend
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1.5, 1, 1], figure=fig)

        # Handle Empty Data
        if self.X_orig.size == 0:
            fig.text(0.5, 0.5, "No Data", ha='center', va='center')
            return fig

        # 1. Apply Rotation to Data
        # We need to center the data first to rotate around centroid
        X_c = (self.X_orig - self.cx) * self.sxy
        Y_c = (self.Y_orig - self.cy) * self.sxy
        Z_c = (self.Z_orig - self.cz) * self.sz

        # Stack
        points = np.stack([X_c, Y_c, Z_c], axis=1) # N x 3

        # Get Rotation Matrix
        R = self._get_rotation_matrix(self.elev, self.azim)

        points_rot = points @ R.T # (3x3 @ 3xN).T -> N x 3

        Xs, Ys, Zs = points_rot[:, 0], points_rot[:, 1], points_rot[:, 2]

        # Common scatter args
        scatter_kwargs = {
            's': self.point_size,
            'alpha': self.alpha,
        }

        if self.is_continuous:
            scatter_kwargs['c'] = self.color
            scatter_kwargs['cmap'] = self.cmap
        elif self.is_categorical:
            scatter_kwargs['c'] = self.c_mapped
        else:
            scatter_kwargs['c'] = 'gray'

        # Global Limits
        lim = self.max_radius
        # Limits: [-lim, lim]

        # --- 3D Plot (Row 0, Span 2) ---
        ax3d = fig.add_subplot(gs[0, :], projection='3d')
        p3d = ax3d.scatter(Xs, Ys, Zs, **scatter_kwargs)

        # Fix Camera
        ax3d.view_init(elev=20, azim=-45)

        ax3d.set_xlim([-lim, lim])
        ax3d.set_ylim([-lim, lim])
        ax3d.set_zlim([-lim, lim])
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')

        # --- Side Projection (Row 1, Left) ---
        # Z vs Y (Y on horizontal, Z on vertical)
        ax_yz = fig.add_subplot(gs[1, 0])
        ax_yz.scatter(Ys, Zs, **scatter_kwargs)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlim([-lim, lim])
        ax_yz.set_ylim([-lim, lim])
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title("Side View (Y-Z)")
        ax_yz.grid(True, linestyle=':', alpha=0.6)

        # --- Front Projection (Row 1, Right) ---
        # Z vs X (X on horizontal, Z on vertical)
        ax_xz = fig.add_subplot(gs[1, 1])
        ax_xz.scatter(Xs, Zs, **scatter_kwargs)
        ax_xz.set_aspect('equal')
        ax_xz.set_xlim([-lim, lim])
        ax_xz.set_ylim([-lim, lim])
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title("Front View (X-Z)")
        ax_xz.grid(True, linestyle=':', alpha=0.6)

        # --- Top Projection (Row 2, Left) ---
        # Y vs X (X horizontal, Y vertical)
        ax_yx = fig.add_subplot(gs[2, 0])
        ax_yx.scatter(Xs, Ys, **scatter_kwargs)
        ax_yx.set_aspect('equal')
        ax_yx.set_xlim([-lim, lim])
        ax_yx.set_ylim([-lim, lim])
        ax_yx.set_xlabel('X')
        ax_yx.set_ylabel('Y')
        ax_yx.set_title("Top View (X-Y)")
        ax_yx.grid(True, linestyle=':', alpha=0.6)

        # --- Legend (Row 2, Right) ---
        ax_leg = fig.add_subplot(gs[2, 1])
        ax_leg.axis('off')

        if self.is_continuous:
            # Safely add colorbar only if collection exists
            if hasattr(p3d, 'cmap'):
                 cbar = plt.colorbar(p3d, ax=ax_leg, fraction=0.8, pad=0.05, aspect=20)
                 cbar.set_label('Value')
        elif self.is_categorical:
            handles = [
                matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=self.cat_color_map[cat],
                                        markersize=10, label=str(cat))
                for cat in self.unique_cats
            ]
            ax_leg.legend(handles=handles, loc='center', title="Categories", frameon=False)

        if self.title:
            fig.suptitle(self.title, fontsize=16)

        fig.tight_layout()
        return fig

def show_iso_scatter(
    X, Y, Z,
    color=None,
    sxy=1, sz=1,
    figsize=(10, 8),
    point_size=5,
    alpha=0.6,
    cmap='viridis',
    max_points=10000,
    title=None
):
    """
    Displays an interactive isometric scatter plot widget.

    Parameters
    ----------
    X, Y, Z : array-like
        Coordinates of points.
    color : array-like, optional
        Color values (continuous or categorical).
    sxy, sz : float
        Scaling factors for XY and Z dimensions.
    figsize : tuple
        Size of the rendered figure.
    point_size : int
        Size of scatter points.
    alpha : float
        Transparency of points.
    cmap : str
        Colormap for continuous data.
    max_points : int
        Maximum number of points to render (random subsampling applied if exceeded).
    title : str, optional
        Plot title.
    """
    return IsoScatterWidget(
        X, Y, Z,
        color=color,
        sxy=sxy,
        sz=sz,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        max_points=max_points,
        title=title
    )
