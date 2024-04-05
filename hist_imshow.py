import numpy as np
import matplotlib.pyplot as plt

def hist_imshow(image, bins=256, return_image_only = False,  **imshow_kwargs):
    """
    Displays an image and its histogram.

    This function processes a given image stack, ensuring it is in a 2D format suitable for display. If the image
    stack has more than two dimensions, the function extracts the middle slice from each of the extra dimensions, 
    and then displays this 2D slice. Alongside the image, the function plots a histogram of the pixel 
    intensities to provide a visual representation of the distribution of pixel values within the image.

    Parameters:
    image (array-like): The input image. Can be multidimensional, but only a 2D slice (from the middle of 
                        any extra dimensions) will be displayed.
    bins (int, optional): The number of bins to use for the histogram. Default is 256.

    Returns:
    matplotlib.figure.Figure: A figure object with two subplots - one showing the image and the other 
                              showing its histogram.

    Note:
    Additional information about the original shape and data type of the image is displayed on the x-axis 
    label of the histogram.

    Example:
    >>> img = np.random.rand(100, 100)  # Generate a random image
    >>> fig = hist_imshow(img)
    >>> plt.show()  # Display the figure with the image and its histogram
    """

    # Ensure 'origin' is set to 'lower' if not specified
    imshow_kwargs.setdefault('origin', 'lower')
    
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

    axes['Image'].imshow(image, **imshow_kwargs)

    # Display histogram
    axes['Histogram'].hist(image.ravel(), bins=bins, density = True, histtype='stepfilled')
    axes['Histogram'].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axes['Histogram'].set_xlabel(f' Pixel intensity \n \n shape: {im_shape} \n dtype: {image.dtype}')
    # axes['Histogram'].set_xlim(0, 1)
    # axes['Histogram'].set_yticks([])


    return fig
