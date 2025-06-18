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
        ax0.imshow(xy, cmap = colormap, vmax=vmax, extent=[0,xdim*sxy,ydim*sxy,0], interpolation = 'nearest')
        ax1.imshow(zy, cmap = colormap, vmax=vmax, extent=[0,zdim*sz,ydim*sxy,0], interpolation = 'nearest')
        ax2.imshow(xz, cmap = colormap, vmax=vmax, extent=[0,xdim*sxy,zdim*sz,0], interpolation = 'nearest')
    else:
        norm=PowerNorm(gamma=gamma, vmax=vmax)
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

    x_slices = slice(*x)
    y_slices = slice(*y)
    z_slices = slice(*z)

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
def show_xyz_max_slice_interactive(im, sxy=1, sz=1, figsize=None, colormap=None, vmin = None, vmax=None, gamma = 1, figsize_scale = 1,  show_crosshair = True, colors = None):
    """
    Display an interactive widget to explore a 3D image by showing a slice in the x, y, and z directions.

    Requires ipywidgets to be installed.

    Parameters
    ----------
    im : array
        3D image to display
    sxy : float
        scaling factor for x and y dimensions
    sz : float
        scaling factor for z dimension
    figsize : tuple
        size of the figure
    colormap : str
        colormap to use
    vmax : float
        maximum value to use for the colormap
    """


    if isinstance(im,list):
        MULTI_CHANNEL = True
        im_shape = im[0].shape
    else:
        im_shape = im.shape


    if sxy!=sz:
        z_xy_ratio=sz/sxy

    if figsize is None:
        z_ , y_, x_ = im_shape
        width_, height_ = x_ + z_ * z_xy_ratio, y_ + z_ * z_xy_ratio
        divisor = max(width_ / 8, height_ / 8)
        width_,  height_ = float(width_ / divisor), float(height_ / divisor)



        
        figsize = (width_, height_)
        if figsize_scale != 1:
            figsize = (width_ * figsize_scale, height_ * figsize_scale)

    def display(x_s,
                y_s,
                z_s,
                x_t,
                y_t,
                z_t,):
        # print(type(x_))

        x_lims = [x_slider.value - x_thick_slider.value,  x_slider.value + x_thick_slider.value]
        y_lims = [y_slider.value - y_thick_slider.value,  y_slider.value + y_thick_slider.value]
        z_lims = [z_slider.value - z_thick_slider.value,  z_slider.value + z_thick_slider.value]

        fig = show_xyz_max_slabs(im, x_lims, y_lims, z_lims, sxy, sz,figsize, colormap, vmin = vmin, vmax = vmax, gamma = gamma, colors = colors)

        if show_crosshair:
            fig.axes[0].axvline(x_lims[0]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[0].axhline(y_lims[0]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[1].axvline(z_lims[0]*sz+0.5*sz, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[1].axhline(y_lims[0]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[2].axvline(x_lims[0]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[2].axhline(z_lims[0]*sz+0.5*sz, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[0].axvline(x_lims[1]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[0].axhline(y_lims[1]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[1].axvline(z_lims[1]*sz+0.5*sz, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[1].axhline(y_lims[1]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[2].axvline(x_lims[1]*sxy+0.5, color='r', linestyle = ':', alpha = 0.3)
            fig.axes[2].axhline(z_lims[1]*sz+0.5*sz, color='r', linestyle = ':', alpha = 0.3)
        plt.show()

    x_thick_slider = IntSlider(min=1, max=im_shape[2]-1, step=1, value=im_shape[2]//64, layout= Layout(width='70%'))
    y_thick_slider = IntSlider(min=1, max=im_shape[1]-1, step=1, value=im_shape[1]//64, layout= Layout(width='70%'))
    z_thick_slider = IntSlider(min=1, max=im_shape[0]-1, step=1, value=im_shape[0]//64, layout= Layout(width='70%'))

    x_slider = IntSlider(min=x_thick_slider.value, max=im_shape[2]-1 - x_thick_slider.value, step=1, value=im_shape[2]//2, layout= Layout(width='70%'))
    y_slider = IntSlider(min=y_thick_slider.value, max=im_shape[1]-1 - y_thick_slider.value, step=1, value=im_shape[1]//2, layout= Layout(width='70%'))
    z_slider = IntSlider(min=z_thick_slider.value, max=im_shape[0]-1 - z_thick_slider.value, step=1, value=im_shape[0]//2, layout= Layout(width='70%'))

    # print(type(x_lims))

    interact(display, x_s = x_slider,
                      y_s = y_slider,
                      z_s = z_slider,
             x_t = x_thick_slider,
             y_t = y_thick_slider,
             z_t = z_thick_slider,)


### New Function
def create_multichannel_rgb(xy_list, xz_list, zy_list, vmin = None, vmax = None, gamma = 1, colors = None):
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

    assert isinstance(xy_list,list)

    num_channels = len(xy_list)

    if gamma == 1:
        gamma = [1] * num_channels
    if vmax is None:
        vmax = [1] * num_channels
    if vmin is None:
        vmin = [0] * num_channels
        
    if colors is None:
        colors = ['magenta', 'cyan', 'yellow', 'green']
        colors = colors[0:num_channels]
    # Convert color names or tuples to RGB
    color_map = [to_rgb(color) for color in colors]


    # Initialize RGB arrays for each orientation
    xy_rgb = np.zeros(xy_list[0].shape + (3,))
    xz_rgb = np.zeros(xz_list[0].shape + (3,))
    zy_rgb = np.zeros(zy_list[0].shape + (3,))

    # Apply PowerNorm per channel
    for idx_i, (xy, xz, zy) in enumerate(zip(xy_list, xz_list, zy_list)):
        xy, xz, zy = xy/xy.max(), xz/xz.max(), zy/zy.max()
        norm = PowerNorm(gamma=gamma[idx_i], vmin=vmin[idx_i], vmax=vmax[idx_i], clip = True)
        xy_norm, xz_norm, zy_norm = norm(xy), norm(xz), norm(zy)  # manually applying norm to the image data
        # xy_list[idx_i], xz_list[idx_i], zy_list[idx_i] = [idx_i]

        # Combine channels into RGB using color weights
        xy_rgb += np.outer(xy_norm.flatten(), color_map[idx_i]).reshape(xy_norm.shape + (3,))
        xz_rgb += np.outer(xz_norm.flatten(), color_map[idx_i]).reshape(xz_norm.shape + (3,))
        zy_rgb += np.outer(zy_norm.flatten(), color_map[idx_i]).reshape(zy_norm.shape + (3,))




    # return show_xyz(xy_rgb, xz_rgb, zy_rgb, vmin = None, vmax=None, gamma = 1, use_plt=True)
    return xy_rgb, xz_rgb, zy_rgb

