# /// script
# requires-python = ">=3.10"
# dependencies = [
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

