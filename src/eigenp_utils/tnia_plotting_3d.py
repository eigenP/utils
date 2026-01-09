# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ipywidgets",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
# ]
# ///
#@title `TNIA plotting functions`

# try:
#     from tnia.plotting.projections import show_xyz_max, show_xyz_slice, show_xyz_max_slabs
# # except:
#     !pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python
#     from tnia.plotting.projections import show_xyz_max, show_xyz_slice, show_xyz_max_slabs

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import gridspec
import numpy as np
from skimage.transform import resize
from matplotlib.colors import PowerNorm, to_rgb, LinearSegmentedColormap


def _norm(arr, symmetric=False, eps=1e-12, dtype=np.float32):
    a = arr.astype(dtype, copy=False)
    if symmetric:
        d = max(eps, float(np.abs(a).max()))
    else:
        d = max(eps, float(a.max()))
    return a / d


# Copyright tnia 2021 - BSD License
def show_xyz_slice(image_to_show, x, y, z, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, use_plt=True):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        x (int): x position of slice
        y (int): y position of slice
        z (int): z position of slice
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """

    slice_zy = np.flip(np.rot90(image_to_show[:,:,x],1),0)
    slice_xz = image_to_show[:,y,:]
    slice_xy = image_to_show[z,:,:]

    return show_xyz(slice_xy, slice_xz, slice_zy, sxy, sz, figsize, colormap, vmax = vmax, vmin = vmin, gamma = gamma, use_plt = use_plt)

# Copyright tnia 2021 - BSD License
def show_xyz_max(image_to_show, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, colors = None):
    """ plots max xy, xz, and zy projections of a 3D image

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size. Defaults to 1.
        sz (float, optional): z pixel size. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """

    return show_xyz_projection(image_to_show, sxy, sz, figsize, np.max, colormap, vmax=vmax, vmin = vmin, gamma = gamma, colors = colors)


def show_xyz_projection(image_to_show, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    projection_y = projector(image_to_show,1)
    projection_x = np.flip(np.rot90(projector(image_to_show,2),1),0)
    projection_z = projector(image_to_show,0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmax=vmax, vmin = vmin, gamma = gamma, colors = colors)

# Copyright tnia 2021 - BSD License
def show_xyz(xy, xz, zy, sxy=1, sz=1,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, use_plt=True, colors = None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    Returns:
        [type]: [description]
    """

    if isinstance(xy,list):
        MULTI_CHANNEL = True
        xy, xz, zy = create_multichannel_rgb(xy, xz, zy, vmin = vmin, vmax=vmax, gamma=gamma, colors = colors)

        # Set those back to default bcs they are dealt with in the RGB function
        vmin, vmax, gamma = None, None, 1


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
    figW, figH = figsize
    hspace_factor = figW / figH
    
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

    if z_xy_ratio!=1:
        xz=resize(xz, (int(xz.shape[0]*z_xy_ratio), xz.shape[1]), preserve_range = True)
        zy=resize(zy, (zy.shape[0], int(zy.shape[1]*z_xy_ratio)), preserve_range = True)


    
    if gamma == 1:
        ax0.imshow(xy, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0], interpolation = 'nearest')
        ax1.imshow(zy, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0], interpolation = 'nearest')
        ax2.imshow(xz, cmap = colormap, vmin=vmin, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0], interpolation = 'nearest')
    else:
        norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
        ax0.imshow(xy, cmap = colormap, norm=norm, extent=[0,xdim*sxy,ydim*sxy,0], interpolation = 'nearest')
        ax1.imshow(zy, cmap = colormap, norm=norm, extent=[0,zdim*sz,ydim*sxy,0], interpolation = 'nearest')
        ax2.imshow(xz, cmap = colormap, norm=norm, extent=[0,xdim*sxy,zdim*sz,0], interpolation = 'nearest')

    ### Axes and titles
    # ax0.set_title('xy')
    # ax1.set_title('zy')
    # ax2.set_title('xz')

    # Remove in-between axes ticks
    for ax in [ax0,ax1,ax2, ax3]:
        ax.axis('off')
    # ax0.xaxis.set_ticklabels([])
    # ax1.yaxis.set_ticklabels([])

    fig.patch.set_alpha(0.01) # set transparent bgnd

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
    fig_h_in = figsize[1]
    fontsize_pt = max(8, min(24, fig_h_in * 72 * 0.03))

    ### Draw
    # center the bar at (x=0.5), y=0.5 in ax3’s normalized coordinates:
    x0 = 0.5 - bar_frac/2
    x1 = 0.5 + bar_frac/2
    y  = 0.5
    
    ax3.hlines(y, x0, x1, transform=ax3.transAxes,
               linewidth=2, color='gray')
    ax3.text(0.5, y - 0.1, f"{int(bar_um)} µm",
             transform=ax3.transAxes,
             ha='center', va='top',
             color='gray',
             fontsize=fontsize_pt)

    return fig



### New function
def show_xyz_max_slabs(image_to_show, x = [0,1], y = [0,1], z = [0,1], sxy=1, sz=1,figsize=(10,10), colormap=None, vmin = None, vmax=None, gamma = 1, colors = None):
    """ plots max xy, xz, and zy projections of a 3D image SLABS (slice intervals)

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        x: slices for x in format [x_1, x_2] where values are integers, to be passed as slice(x_1, x_2, None)
        y: slices for y in format [y_1, y_2] where values are integers
        z: slices for z in format [z_1, z_2] where values are integers
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """
    ### Coerce into integers for slices
    x_ = [int(i) for i in x]
    y_ = [int(i) for i in y]
    z_ = [int(i) for i in z]

    x_slices = slice(*x_)
    y_slices = slice(*y_)
    z_slices = slice(*z_)

    return show_xyz_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy, sz, figsize, np.max, colormap, vmax = vmax, vmin = vmin, gamma = gamma, colors = colors)


### New function
def show_xyz_projection_slabs(image_to_show, x_slices, y_slices, z_slices, sxy=1, sz=1,figsize=(10,10), projector=np.max, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None):
    """ generates xy, xz, and zy max projections of a 3D image and plots them

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show (3d numpy array): image to plot
        sxy (float, optional): xy pixel size of 3D. Defaults to 1.
        sz (float, optional): z pixel size of 3D. Defaults to 1.
        figsize (tuple): size of figure to
        projector: function to project with
        colormap (_type_, optional): _description_. Defaults to None.
        vmax (float, optional): maximum value for display range. Defaults to None.
    """

    if isinstance(image_to_show, list):
        images_to_show_list = image_to_show
        projection_y = [] 
        projection_x = [] 
        projection_z = []
        for image_to_show in images_to_show_list:
            projection_y.append(projector(image_to_show[:,y_slices,:],1))
            projection_x.append(np.flip(np.rot90(projector(image_to_show[:,:,x_slices],2),1),0))
            projection_z.append(projector(image_to_show[z_slices,:,:],0))
    else:
        projection_y = projector(image_to_show[:,y_slices,:],1)
        projection_x = np.flip(np.rot90(projector(image_to_show[:,:,x_slices],2),1),0)
        projection_z = projector(image_to_show[z_slices,:,:],0)

    return show_xyz(projection_z, projection_y, projection_x, sxy, sz, figsize, colormap, vmax = vmax, vmin = vmin, gamma = gamma, colors = colors)


from ipywidgets import interact, interactive, IntSlider, FloatRangeSlider, Layout, Text, Button, VBox, HBox
import functools
import warnings

def add_save_ui(func):
    """
    Decorator that wraps a function returning an ipywidgets.interactive object.
    It appends a filename text box and a 'Save as SVG' button to the UI.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # The wrapped function must return an `interactive` widget
        w = func(*args, **kwargs)

        # Create UI elements for saving
        fname_box = Text(value='filepath_save.svg', description='Filename:')
        save_btn = Button(description='Save as SVG')

        def on_save_click(b):
            # The figure is stored in w.result
            fig = w.result
            if fig is None:
                print("No figure to save yet.")
                return

            filename = fname_box.value
            # Ensure proper extension if not present? Or just trust user?
            # User said "suggest filepath_save.svg"

            try:
                fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
                print(f"Saved to {filename}")
            except Exception as e:
                print(f"Error saving file: {e}")

        save_btn.on_click(on_save_click)

        # Return a container with the original widget + save controls
        # Using VBox to stack them
        ui = VBox([w, HBox([fname_box, save_btn])])
        return ui

    return wrapper


### New function

@add_save_ui
def show_xyz_max_slice_interactive(
    im,
    sxy=1, sz=1,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True,
    colors=None,
    # NEW optional initial values (centers and half-thicknesses)
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    """
    Display an interactive widget to explore a 3D image by showing a slice
    in the x, y, and z directions.

    .. warning::
        This function is deprecated. Please use the version from
        `eigenp_utils.tnia_plotting_anywidgets` instead.

    Parameters
    ----------
    im : array or list[array]
        3D image (Z, Y, X) or list of channels (each Z, Y, X)
    sxy, sz : float
        voxel sizes in XY and Z
    figsize : tuple or None
        figure size; if None, computed from data size
    colormap : str or None
    vmin, vmax : float or None
    gamma : float
    figsize_scale : float
    show_crosshair : bool
    colors : list[str] or None
    x_s, y_s, z_s : int or None
        OPTIONAL initial centers (X, Y, Z). If None, defaults to midpoints.
    x_t, y_t, z_t : int or None
        OPTIONAL initial half-thicknesses in voxels. If None, defaults to ~size/64.
        NOTE: Sliders remain interactive and can override these values.
    """

    warnings.warn(
        "show_xyz_max_slice_interactive is deprecated. "
        "Please import and use the version from eigenp_utils.tnia_plotting_anywidgets instead. "
        "Example: from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive",
        DeprecationWarning,
        stacklevel=2
    )

    # Support multi-channel input
    im_shape = (im[0].shape if isinstance(im, list) else im.shape)  # (Z, Y, X)
    Z, Y, X = im_shape

    # Compute z/xy aspect ratio
    z_xy_ratio = (sz / sxy) if sxy != sz else 1

    # Auto figsize if not provided
    if figsize is None:
        width_px  = X + Z * z_xy_ratio
        height_px = Y + Z * z_xy_ratio
        # keep visual scale stable across sizes
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    # ---- Defaults (clip to valid ranges) ----
    def _clip(v, lo, hi):
        return int(max(lo, min(hi, v)))

    # Sensible default half-thickness ~ size/64 (at least 1)
    def _default_t(n):  # n is dimension length
        return max(1, n // 64)

    # Resolve initial thicknesses (half-widths)
    x_t0 = _clip(x_t if x_t is not None else _default_t(X), 1, max(1, X - 1))
    y_t0 = _clip(y_t if y_t is not None else _default_t(Y), 1, max(1, Y - 1))
    z_t0 = _clip(z_t if z_t is not None else _default_t(Z), 1, max(1, Z - 1))

    # Given t, valid center ranges are [t, dim-1 - t]
    def _center_bounds(dim, t):
        lo, hi = t, max(t, dim - 1 - t)
        return lo, hi

    x_lo0, x_hi0 = _center_bounds(X, x_t0)
    y_lo0, y_hi0 = _center_bounds(Y, y_t0)
    z_lo0, z_hi0 = _center_bounds(Z, z_t0)

    # Resolve initial centers (defaults to midpoints if not provided)
    x_s0 = _clip(x_s if x_s is not None else X // 2, x_lo0, x_hi0)
    y_s0 = _clip(y_s if y_s is not None else Y // 2, y_lo0, y_hi0)
    z_s0 = _clip(z_s if z_s is not None else Z // 2, z_lo0, z_hi0)

    # ---- Sliders ----
    # Thickness sliders first (so position slider ranges can depend on them)
    x_thick_slider = IntSlider(min=1, max=max(1, X - 1), step=1, value=x_t0, layout=Layout(width='70%'))
    y_thick_slider = IntSlider(min=1, max=max(1, Y - 1), step=1, value=y_t0, layout=Layout(width='70%'))
    z_thick_slider = IntSlider(min=1, max=max(1, Z - 1), step=1, value=z_t0, layout=Layout(width='70%'))

    # Position sliders with ranges tied to initial thicknesses
    x_slider = IntSlider(min=x_lo0, max=x_hi0, step=1, value=x_s0, layout=Layout(width='70%'))
    y_slider = IntSlider(min=y_lo0, max=y_hi0, step=1, value=y_s0, layout=Layout(width='70%'))
    z_slider = IntSlider(min=z_lo0, max=z_hi0, step=1, value=z_s0, layout=Layout(width='70%'))

    # Keep position slider bounds consistent if thickness changes interactively
    def _bind_bounds(thick_sl, pos_sl, dim):
        def _on_change(change):
            t = change["new"]
            lo = t
            hi = max(t, dim - 1 - t)
            pos_sl.min = lo
            pos_sl.max = hi
            # clamp current value if now out of bounds
            if pos_sl.value < lo: pos_sl.value = lo
            if pos_sl.value > hi: pos_sl.value = hi
        thick_sl.observe(_on_change, names='value')

    _bind_bounds(x_thick_slider, x_slider, X)
    _bind_bounds(y_thick_slider, y_slider, Y)
    _bind_bounds(z_thick_slider, z_slider, Z)

    # ---- Display callback ----
    def _display(_x_s, _y_s, _z_s, _x_t, _y_t, _z_t):
        # Build current limits from sliders (centers +/- half-thickness)
        x_lims = [x_slider.value - x_thick_slider.value, x_slider.value + x_thick_slider.value]
        y_lims = [y_slider.value - y_thick_slider.value, y_slider.value + y_thick_slider.value]
        z_lims = [z_slider.value - z_thick_slider.value, z_slider.value + z_thick_slider.value]

        fig = show_xyz_max_slabs(
            im, x_lims, y_lims, z_lims,
            sxy=sxy, sz=sz, figsize=figsize, colormap=colormap,
            vmin=vmin, vmax=vmax, gamma=gamma, colors=colors
        )

        if show_crosshair:
            # XY
            fig.axes[0].axvline(x_lims[0]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[0]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axvline(x_lims[1]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[1]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            # ZY
            fig.axes[1].axvline(z_lims[0]*sz + 0.5*sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[0]*sxy + 0.5,     color='r', ls=':', alpha=0.3)
            fig.axes[1].axvline(z_lims[1]*sz + 0.5*sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[1]*sxy + 0.5,     color='r', ls=':', alpha=0.3)
            # XZ
            fig.axes[2].axvline(x_lims[0]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[0]*sz + 0.5*sz, color='r', ls=':', alpha=0.3)
            fig.axes[2].axvline(x_lims[1]*sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[1]*sz + 0.5*sz, color='r', ls=':', alpha=0.3)

        # plt.show()
        return fig

    return interactive(
        _display,
        _x_s=x_slider, _y_s=y_slider, _z_s=z_slider,
        _x_t=x_thick_slider, _y_t=y_thick_slider, _z_t=z_thick_slider
    )


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
    vmin=None, vmax=None, gamma=1, colors=None,
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
    blend : str
        'add' (default), 'screen', or 'max'.
    soft_clip : bool
        For 'add' mode, compress values >1 instead of hard clipping.
    """

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

    if colors is None:
        colors = ['magenta', 'cyan', 'yellow', 'green'][:n]
    color_map = [np.asarray(to_rgb(c), dtype=np.float32) for c in colors]

    # Determine per-channel vmin/vmax if not provided
    if vmin is None:
        vmins = [0.0] * n
    else:
        vmins = list(vmin) if isinstance(vmin, (list, tuple)) else [float(vmin)] * n

    if vmax is None:
        vmaxs = []
        for xy, xz, zy in zip(xy_list, xz_list, zy_list):
            # global max across orientations for that channel
            vmaxs.append(float(max(np.max(xy), np.max(xz), np.max(zy))))
    else:
        vmaxs = list(vmax) if isinstance(vmax, (list, tuple)) else [float(vmax)] * n

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
        out = np.clip(out, 0.0, 1.0)
        return out if g == 1 else np.power(out, g, dtype=np.float32)

    # Per-channel accumulate
    for i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
        c = color_map[i]  # (3,)
        g = gammas[i]
        lo, hi = vmins[i], vmaxs[i]

        xy_n = _norm(xy, lo, hi, g)[..., None] * c  # (H,W,3)
        xz_n = _norm(xz, lo, hi, g)[..., None] * c
        zy_n = _norm(zy, lo, hi, g)[..., None] * c

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

    # # return show_xyz(xy_rgb, xz_rgb, zy_rgb, vmin = None, vmax=None, gamma = 1, use_plt=True)
    # return xy_rgb, xz_rgb, zy_rgb


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
        norm = (arr - vmin[c]) / max(1e-9, vmax[c] - vmin[c])
        norm = np.clip(norm, 0, 1)
        if gammas[c] != 1:
            norm = norm**gammas[c]
        rgb = np.asarray(to_rgb(base_colors[c]))
        colors += norm[:, None] * rgb

    if soft_clip:
        maxval = colors.max(axis=1, keepdims=True)
        scale = np.maximum(1.0, maxval)
        colors = colors / scale

    return np.clip(colors, 0, 1)

@add_save_ui
def show_xyz_max_scatter_interactive(
    X, Y, Z,
    channels=None,                 # None | int-array (IDs) | float-array (continuous) | list of arrays (IDs or continuous)
    sxy=1, sz=1,
    render='points',              # 'density' or 'points'
    bins=512,                      # int or (nx, ny) for density mode
    point_size=4, alpha=0.6,       # points mode
    colors=None,                   # per-channel base colors; for single continuous, colors[0] is the target color
    gamma=1, vmin=None, vmax=None, # tone map for density & multi-cont blending
    figsize=None, figsize_scale=1.0,
    show_crosshair=True,
    # optional initial centers and half-thicknesses (voxels)
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    warnings.warn(
        "show_xyz_max_scatter_interactive is deprecated. "
        "Please import and use the version from eigenp_utils.tnia_plotting_anywidgets instead. "
        "Example: from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_scatter_interactive",
        DeprecationWarning,
        stacklevel=2
    )

    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider, Layout
    from matplotlib.colors import to_rgb

    # ---------- Input normalization ----------
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
    N = len(X)
    assert X.shape == Y.shape == Z.shape, "X, Y, Z must have same length"

    # Channel mode detection
    def _is_int_like(a): return np.issubdtype(np.asarray(a).dtype, np.integer) or np.asarray(a).dtype == bool
    def _as_1d(a):
        a = np.asarray(a)
        return a.reshape(-1) if a.ndim == 1 else a

    mode = 'single'        # default when channels is None
    ch_ids = None          # for discrete IDs
    cont_single = None     # for single continuous
    cont_multi = None      # for multi continuous (N,C) array
    idx_lists = None       # list of index arrays (discrete)

    if channels is None:
        mode = 'single'  # one channel, categorical color
    elif isinstance(channels, (list, tuple)):
        # list: either index arrays (discrete) OR float arrays (continuous multi)
        arrs = [ _as_1d(a) for a in channels ]
        lens = {len(a) for a in arrs}
        if lens == {N} and any(np.issubdtype(a.dtype, np.floating) for a in arrs):
            mode = 'cont_multi'   # multiple continuous channels, length N each
            cont_multi = np.stack([a.astype(float) for a in arrs], axis=1)  # (N, C)
        else:
            mode = 'idx_lists'    # list of index arrays
            idx_lists = [np.asarray(ix, dtype=int) for ix in channels]
    else:
        a = _as_1d(channels)
        if _is_int_like(a):
            mode = 'ids'          # discrete IDs per point
            ch_ids = np.asarray(a, dtype=int)
            assert ch_ids.shape == (N,), "channels IDs must be length N"
        else:
            mode = 'cont_single'  # single continuous scalar per point
            cont_single = np.asarray(a, dtype=float)
            assert cont_single.shape == (N,), "continuous channels must be length N"

    # Channel count & default colors
    if mode in ('single',):
        C = 1
    elif mode == 'ids':
        C = int(ch_ids.max()) + 1 if ch_ids.size else 1
    elif mode == 'idx_lists':
        C = len(idx_lists)
    elif mode == 'cont_single':
        C = 1
    else:  # 'cont_multi'
        C = cont_multi.shape[1]

    if colors is None:
        default_cols = ['magenta', 'cyan', 'yellow', 'green', 'red', 'lime', 'blue', 'orange']
        colors = default_cols[:max(1, C)]
    colors_rgb = [to_rgb(c) for c in colors]

    # ---------- Bounds & layout ----------
    xmin, xmax = float(np.floor(X.min())), float(np.ceil(X.max()))
    ymin, ymax = float(np.floor(Y.min())), float(np.ceil(Y.max()))
    zmin, zmax = float(np.floor(Z.min())), float(np.ceil(Z.max()))

    XN = xmax - xmin + 1
    YN = ymax - ymin + 1
    ZN = zmax - zmin + 1
    z_xy_ratio = (sz / sxy) if sxy != sz else 1

    if figsize is None:
        width_px  = XN + ZN * z_xy_ratio
        height_px = YN + ZN * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, int(n // 64))
    Xdim = int(np.ceil(XN)); Ydim = int(np.ceil(YN)); Zdim = int(np.ceil(ZN))

    x_t0 = int(x_t if x_t is not None else _default_t(Xdim))
    y_t0 = int(y_t if y_t is not None else _default_t(Ydim))
    z_t0 = int(z_t if z_t is not None else _default_t(Zdim))

    def _clip(v, lo, hi): return int(max(lo, min(hi, v)))
    x_center_default = int(np.round((xmin + xmax) * 0.5))
    y_center_default = int(np.round((ymin + ymax) * 0.5))
    z_center_default = int(np.round((zmin + zmax) * 0.5))

    def _bounds(lo, hi, t):
        lo_c = int(np.ceil(lo + t))
        hi_c = int(np.floor(hi - t))
        if hi_c < lo_c: hi_c = lo_c
        return lo_c, hi_c

    x_lo0, x_hi0 = _bounds(xmin, xmax, x_t0)
    y_lo0, y_hi0 = _bounds(ymin, ymax, y_t0)
    z_lo0, z_hi0 = _bounds(zmin, zmax, z_t0)

    x_s0 = _clip(int(x_s if x_s is not None else x_center_default), x_lo0, x_hi0)
    y_s0 = _clip(int(y_s if y_s is not None else y_center_default), y_lo0, y_hi0)
    z_s0 = _clip(int(z_s if z_s is not None else z_center_default), z_lo0, z_hi0)

    # Sliders
    from ipywidgets import interact, IntSlider, Layout
    x_thick_slider = IntSlider(min=1, max=max(1, Xdim - 1), step=1, value=int(x_t0), layout=Layout(width='70%'))
    y_thick_slider = IntSlider(min=1, max=max(1, Ydim - 1), step=1, value=int(y_t0), layout=Layout(width='70%'))
    z_thick_slider = IntSlider(min=1, max=max(1, Zdim - 1), step=1, value=int(z_t0), layout=Layout(width='70%'))

    x_slider = IntSlider(min=x_lo0, max=x_hi0, step=1, value=x_s0, layout=Layout(width='70%'))
    y_slider = IntSlider(min=y_lo0, max=y_hi0, step=1, value=y_s0, layout=Layout(width='70%'))
    z_slider = IntSlider(min=z_lo0, max=z_hi0, step=1, value=z_s0, layout=Layout(width='70%'))

    def _bind_bounds(thick_sl, pos_sl, lo, hi):
        def _on_change(change):
            t = int(change["new"])
            lo_c, hi_c = _bounds(lo, hi, t)
            pos_sl.min, pos_sl.max = lo_c, hi_c
            if pos_sl.value < lo_c: pos_sl.value = lo_c
            if pos_sl.value > hi_c: pos_sl.value = hi_c
        thick_sl.observe(_on_change, names='value')
    _bind_bounds(x_thick_slider, x_slider, xmin, xmax)
    _bind_bounds(y_thick_slider, y_slider, ymin, ymax)
    _bind_bounds(z_thick_slider, z_slider, zmin, zmax)

    # ---------- Density helpers ----------
    def _resolve_bins(B):
        if isinstance(B, (tuple, list)) and len(B) == 2:
            bx, by = int(B[0]), int(B[1])
        else:
            bx = by = int(B)
        bx = max(1, bx); by = max(1, by)
        return bx, by

    BX, BY = _resolve_bins(bins)  # cols (X/Z), rows (Y)

    def _hist2d(x, y, xr, yr, w=None):
        # rows = BY (y-bins), cols = BX (x-bins)
        H, _, _ = np.histogram2d(y, x, bins=[BY, BX], range=[yr, xr], weights=w)
        return H

    EMPTY_XY = np.zeros((BY, BX), dtype=float)
    EMPTY_XZ = np.zeros((BY, BX), dtype=float)
    EMPTY_ZY = np.zeros((BY, BX), dtype=float)

    # ---------- Render callback ----------
    def _display(_x_s, _y_s, _z_s, _x_t, _y_t, _z_t):
        x_lims = (x_slider.value - x_thick_slider.value, x_slider.value + x_thick_slider.value)
        y_lims = (y_slider.value - y_thick_slider.value, y_slider.value + y_thick_slider.value)
        z_lims = (z_slider.value - z_thick_slider.value, z_slider.value + z_thick_slider.value)

        if render == 'density':
            # Prepare per-orientation channel stacks
            xy_list, xz_list, zy_list = [], [], []

            if mode in ('single', 'ids', 'idx_lists'):
                # Discrete channels (counts)
                if mode == 'single':
                    idx_lists_local = [np.arange(N)]
                elif mode == 'ids':
                    idx_lists_local = [np.nonzero(ch_ids == c)[0] for c in range(C)]
                else:
                    idx_lists_local = idx_lists

                for idxs in idx_lists_local:
                    if idxs.size == 0:
                        xy_list.append(EMPTY_XY.copy()); xz_list.append(EMPTY_XZ.copy()); zy_list.append(EMPTY_ZY.copy()); continue
                    Xi, Yi, Zi = X[idxs], Y[idxs], Z[idxs]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    Hxy = _hist2d(Xi[mZ], Yi[mZ], (xmin, xmax+1), (ymin, ymax+1)) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(Xi[mY], Zi[mY], (xmin, xmax+1), (zmin, zmax+1)) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(Zi[mX], Yi[mX], (zmin, zmax+1), (ymin, ymax+1)) if mX.any() else EMPTY_ZY.copy()
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            elif mode == 'cont_single':
                # Single continuous → weighted histograms; map to black→color
                vals = cont_single
                mZ = (Z >= z_lims[0]) & (Z <= z_lims[1])
                mY = (Y >= y_lims[0]) & (Y <= y_lims[1])
                mX = (X >= x_lims[0]) & (X <= x_lims[1])

                Hxy = _hist2d(X[mZ], Y[mZ], (xmin, xmax+1), (ymin, ymax+1), w=vals[mZ]) if mZ.any() else EMPTY_XY.copy()
                Hxz = _hist2d(X[mY], Z[mY], (xmin, xmax+1), (zmin, zmax+1), w=vals[mY]) if mY.any() else EMPTY_XZ.copy()
                Hzy = _hist2d(Z[mX], Y[mX], (zmin, zmax+1), (ymin, ymax+1), w=vals[mX]) if mX.any() else EMPTY_ZY.copy()

                xy_list = [Hxy]; xz_list = [Hxz]; zy_list = [Hzy]
                if colors is None:  # default continuous color
                    colors_use = ['cyan']
                else:
                    colors_use = colors
            else:
                # cont_multi: multiple continuous channels → weighted hist per channel
                vals = cont_multi  # (N, C)
                for c in range(C):
                    v = vals[:, c]
                    mZ = (Z >= z_lims[0]) & (Z <= z_lims[1])
                    mY = (Y >= y_lims[0]) & (Y <= y_lims[1])
                    mX = (X >= x_lims[0]) & (X <= x_lims[1])

                    Hxy = _hist2d(X[mZ], Y[mZ], (xmin, xmax+1), (ymin, ymax+1), w=v[mZ]) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(X[mY], Z[mY], (xmin, xmax+1), (zmin, zmax+1), w=v[mY]) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(Z[mX], Y[mX], (zmin, zmax+1), (ymin, ymax+1), w=v[mX]) if mX.any() else EMPTY_ZY.copy()

                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)
                colors_use = colors  # must have length C

            # Combine channels → RGB images
            if mode in ('single', 'ids', 'idx_lists', 'cont_multi'):
                colors_for_rgb = colors if colors is not None else ['magenta', 'cyan', 'yellow', 'green'][:len(xy_list)]
            else:  # cont_single
                colors_for_rgb = colors_use

            xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
                xy_list, xz_list, zy_list,
                vmin=vmin, vmax=vmax, gamma=gamma, colors=colors_for_rgb, blend='add', soft_clip=True
            )

            fig = show_xyz(
                xy_rgb, xz_rgb, zy_rgb,
                sxy=sxy, sz=sz, figsize=figsize, colormap=None,
                vmin=None, vmax=None, gamma=1, use_plt=True, colors=None
            )

            # Black background (opaque)
            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("black")
            for ax in fig.axes:
                ax.set_facecolor("black")

        else:
            # ---------- POINTS mode: use plt.subplots (your preference) ----------
            # Build 2x2 with custom ratios
            width_ratios  = [int(xmax - xmin + 1), int((zmax - zmin + 1) * z_xy_ratio)]
            height_ratios = [int(ymax - ymin + 1), int((zmax - zmin + 1) * z_xy_ratio)]

            fig, axs = plt.subplots(
                2, 2, figsize=figsize, constrained_layout=False,
                gridspec_kw=dict(width_ratios=width_ratios, height_ratios=height_ratios),
                facecolor='black'
            )
            axXY, axZY = axs[0,0], axs[0,1]
            axXZ, axBar = axs[1,0], axs[1,1]
            for ax in (axXY, axZY, axXZ, axBar):
                ax.set_facecolor('black'); ax.axis('off')

            # Masks for slabs
            mZ_all = (Z >= z_lims[0]) & (Z <= z_lims[1])
            mY_all = (Y >= y_lims[0]) & (Y <= y_lims[1])
            mX_all = (X >= x_lims[0]) & (X <= x_lims[1])

            if mode == 'cont_single':
                # Single scalar → black→color gradient
                vals = cont_single
                cmap = black_to(colors[0] if colors else 'cyan')
                norm = plt.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))

                if mZ_all.any():
                    axXY.scatter(X[mZ_all]*sxy, Y[mZ_all]*sxy, c=vals[mZ_all], cmap=cmap, norm=norm,
                                 s=point_size, alpha=alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(X[mY_all]*sxy, Z[mY_all]*sz,  c=vals[mY_all], cmap=cmap, norm=norm,
                                 s=point_size, alpha=alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(Z[mX_all]*sz,  Y[mX_all]*sxy, c=vals[mX_all], cmap=cmap, norm=norm,
                                 s=point_size, alpha=alpha, linewidths=0)

            elif mode == 'cont_multi':
                # Multiple scalar channels → blended RGB per point
                cols = blend_colors(cont_multi, colors, vmin=vmin, vmax=vmax, gamma=gamma, soft_clip=True)

                if mZ_all.any():
                    axXY.scatter(X[mZ_all]*sxy, Y[mZ_all]*sxy, c=cols[mZ_all], s=point_size, alpha=alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(X[mY_all]*sxy, Z[mY_all]*sz,  c=cols[mY_all], s=point_size, alpha=alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(Z[mX_all]*sz,  Y[mX_all]*sxy, c=cols[mX_all], s=point_size, alpha=alpha, linewidths=0)

            else:
                # Discrete channels (single/ids/idx_lists) → per-channel solid color
                if mode == 'single':
                    idx_lists_local = [np.arange(N)]
                elif mode == 'ids':
                    idx_lists_local = [np.nonzero(ch_ids == c)[0] for c in range(C)]
                else:
                    idx_lists_local = idx_lists

                for c, idxs in enumerate(idx_lists_local):
                    if idxs.size == 0: continue
                    Xi, Yi, Zi = X[idxs], Y[idxs], Z[idxs]
                    col = [colors_rgb[c % len(colors_rgb)]]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    if mZ.any(): axXY.scatter(Xi[mZ]*sxy, Yi[mZ]*sxy, s=point_size, c=col, alpha=alpha, linewidths=0)
                    if mY.any(): axXZ.scatter(Xi[mY]*sxy, Zi[mY]*sz,  s=point_size, c=col, alpha=alpha, linewidths=0)
                    if mX.any(): axZY.scatter(Zi[mX]*sz,  Yi[mX]*sxy, s=point_size, c=col, alpha=alpha, linewidths=0)

            # Axis limits (physical units)
            axXY.set_xlim([xmin*sxy, (xmax+1)*sxy]); axXY.set_ylim([(ymax+1)*sxy, ymin*sxy])
            axXZ.set_xlim([xmin*sxy, (xmax+1)*sxy]); axXZ.set_ylim([(zmax+1)*sz,  zmin*sz ])
            axZY.set_xlim([zmin*sz,  (zmax+1)*sz ]); axZY.set_ylim([(ymax+1)*sxy, ymin*sxy])

            # Crosshair boxes (optional)
            if show_crosshair:
                axXY.vlines([x_lims[0]*sxy, (x_lims[1]+1)*sxy], ymin*sxy, (ymax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
                axXY.hlines([y_lims[0]*sxy, (y_lims[1]+1)*sxy], xmin*sxy, (xmax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.vlines([z_lims[0]*sz,  (z_lims[1]+1)*sz],  ymin*sxy, (ymax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.hlines([y_lims[0]*sxy, (y_lims[1]+1)*sxy], zmin*sz,   (zmax+1)*sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.vlines([x_lims[0]*sxy, (x_lims[1]+1)*sxy], zmin*sz,   (zmax+1)*sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.hlines([z_lims[0]*sz,  (z_lims[1]+1)*sz],  xmin*sxy,  (xmax+1)*sxy, colors='r', linestyles=':', alpha=0.3)

            # Scale bar (kept opaque)
            fig.patch.set_alpha(1.0)
            width_um = (xmax - xmin + 1) * sxy
            target = width_um * 0.2
            def nice_length(x):
                exp = np.floor(np.log10(x))
                for m in [5,2,1]:
                    val = m * 10**exp
                    if val <= x: return val
                return x
            bar_um = nice_length(target)
            bar_pix = bar_um / sxy
            bar_frac = bar_pix / (xmax - xmin + 1)
            fig_h_in = figsize[1]
            fontsize_pt = max(8, min(24, fig_h_in * 72 * 0.03))
            x0 = 0.5 - bar_frac/2; x1 = 0.5 + bar_frac/2; y = 0.5
            axBar.hlines(y, x0, x1, transform=axBar.transAxes, linewidth=2, color='gray')
            axBar.text(0.5, y - 0.1, f"{int(bar_um)} µm", transform=axBar.transAxes,
                       ha='center', va='top', color='gray', fontsize=fontsize_pt)

            fig.tight_layout(pad=0.0)

        # Crosshair overlays for density (after show_xyz)
        if render == 'density' and show_crosshair:
            axXY, axZY, axXZ = fig.axes[0], fig.axes[1], fig.axes[2]
            axXY.vlines([x_lims[0]*sxy + 0.5, (x_lims[1]+1)*sxy + 0.5], ymin*sxy, (ymax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
            axXY.hlines([y_lims[0]*sxy + 0.5, (y_lims[1]+1)*sxy + 0.5], xmin*sxy, (xmax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
            axZY.vlines([z_lims[0]*sz + 0.5*sz, (z_lims[1]+1)*sz + 0.5*sz], ymin*sxy, (ymax+1)*sxy, colors='r', linestyles=':', alpha=0.3)
            axZY.hlines([y_lims[0]*sxy + 0.5,       (y_lims[1]+1)*sxy + 0.5], zmin*sz, (zmax+1)*sz, colors='r', linestyles=':', alpha=0.3)
            axXZ.vlines([x_lims[0]*sxy + 0.5, (x_lims[1]+1)*sxy + 0.5], zmin*sz, (zmax+1)*sz, colors='r', linestyles=':', alpha=0.3)
            axXZ.hlines([z_lims[0]*sz + 0.5*sz, (z_lims[1]+1)*sz + 0.5*sz], xmin*sxy, (xmax+1)*sxy, colors='r', linestyles=':', alpha=0.3)

        # Ensure opaque black everywhere
        fig.patch.set_alpha(1.0)
        fig.patch.set_facecolor('black')
        for ax in fig.axes:
            ax.set_facecolor('black')

        # plt.show()
        return fig

    return interactive(
        _display,
        _x_s=x_slider, _y_s=y_slider, _z_s=z_slider,
        _x_t=x_thick_slider, _y_t=y_thick_slider, _z_t=z_thick_slider
    )
