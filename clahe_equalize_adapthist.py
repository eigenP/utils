import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity



def clahe(image, clip_limit = 0.02):
    # Remove singleton dimensions
    image = np.squeeze(image)
    dtype_ = str(image.dtype)

    # Expects XY Z(T) so need to transpose
    image = np.moveaxis(image, 0,-1)

    # # Rescale intensitites to 0-1 for CLAHE to work
    # image = rescale_intensity(image, out_range = (0,1))

    # Check the number of dimensions of the input image
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim > 3:
        raise ValueError("The input image has too many dimensions! It must be 2D or 3D after squeezing.")

    # # Do the CLAHE
    # Use explicit kernel_size to trick CLAHE into doing it in 2D + t vs 3D
    kernel_size = (image.shape[0] // 8,
                   image.shape[1] // 8,
                   1)
    kernel_size = np.array(kernel_size)


    data = equalize_adapthist(image, kernel_size = kernel_size, clip_limit = clip_limit)
    # Rescale intensity
    result_image = rescale_intensity(data, out_range = dtype_)


    # Re-transpose to return to original dim order
    result_image = np.moveaxis(result_image, -1,0)


    return np.squeeze(result_image)
