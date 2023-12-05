import numpy as np
import matplotlib.pyplot as plt

def hist_imshow(image, bins=256):

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
        print(slices)
        image = image[slices]

    fig, axes = plt.subplot_mosaic([['Image', '.'], ['Image', 'Histogram'], ['Image', '.']],
                                   layout = 'constrained')

    axes['Image'].imshow(image)

    # Display histogram
    axes['Histogram'].hist(image.ravel(), bins=bins, density = True, histtype='stepfilled')
    axes['Histogram'].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axes['Histogram'].set_xlabel(f' Pixel intensity \n \n shape: {im_shape} \n dtype: {image.dtype}')
    # axes['Histogram'].set_xlim(0, 1)
    # axes['Histogram'].set_yticks([])


    return fig
