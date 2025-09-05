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
from matplotlib.colors import PowerNorm, to_rgb


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
        vmax, gamma = None, 1


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


from ipywidgets import interact, IntSlider, FloatRangeSlider, Layout


### New function

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

        plt.show()

    interact(
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


def create_multichannel_rgb(xy_list, xz_list, zy_list, vmin=None, vmax=None, gamma=1, colors=None):
    """
    Display an interactive widget to explore a 3D image by showing a slice in the x, y, and z directions.

    Requires ipywidgets to be installed.

    Parameters
    ----------
    xy_list, xz_list, zy_list : lists of images (len of list is number of channels)
    vmax : float
        maximum value to use for the PowerNorm
    gamma : float
        gamma value to use for the PowerNorm
    colors : list of strs
        one color per channel
    """
    assert isinstance(xy_list, list) and isinstance(xz_list, list) and isinstance(zy_list, list)
    n = len(xy_list)
    assert len(xz_list) == n and len(zy_list) == n, "xy/xz/zy must have same number of channels"

    # Allocate composite RGB buffers for each orientation
    xy_rgb = np.zeros(xy_list[0].shape + (3,), dtype=np.float32)
    xz_rgb = np.zeros(xz_list[0].shape + (3,), dtype=np.float32)
    zy_rgb = np.zeros(zy_list[0].shape + (3,), dtype=np.float32)

    # Prepare per-channel params
    if isinstance(gamma, (list, tuple)): gammas = list(gamma)
    else:                                 gammas = [gamma] * n
    if vmin is None: vmins = [0.0] * n
    else:            vmins = list(vmin) if isinstance(vmin, (list, tuple)) else [vmin] * n
    if vmax is None: vmaxs = [1.0] * n
    else:            vmaxs = list(vmax) if isinstance(vmax, (list, tuple)) else [vmax] * n

    # Default colors if none provided
    if colors is None:
        colors = ['magenta', 'cyan', 'yellow', 'green'][:n]
    color_map = [np.asarray(to_rgb(c), dtype=np.float32) for c in colors]

    # Normalize helper (handles ints; avoids in-place ops)
    def _to_float(a):  # safe upcast
        return a.astype(np.float32, copy=False)

    for i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
        # Per-channel PowerNorm (respects vmin/vmax/gamma)
        norm = PowerNorm(gamma=gammas[i], vmin=vmins[i], vmax=vmaxs[i], clip=True)

        xy_n = norm(_to_float(xy))
        xz_n = norm(_to_float(xz))
        zy_n = norm(_to_float(zy))

        c = color_map[i]  # shape (3,)

        # Broadcast add: (H,W) -> (H,W,1) * (3,)
        xy_rgb += xy_n[..., None] * c
        xz_rgb += xz_n[..., None] * c
        zy_rgb += zy_n[..., None] * c

    # Keep display-safe range
    xy_rgb = np.clip(xy_rgb, 0, 1)
    xz_rgb = np.clip(xz_rgb, 0, 1)
    zy_rgb = np.clip(zy_rgb, 0, 1)

    return xy_rgb, xz_rgb, zy_rgb

    # # return show_xyz(xy_rgb, xz_rgb, zy_rgb, vmin = None, vmax=None, gamma = 1, use_plt=True)
    # return xy_rgb, xz_rgb, zy_rgb

