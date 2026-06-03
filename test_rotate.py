import numpy as np
from skimage.transform import rotate

img = np.zeros((5, 5))
img[0, 2] = 1 # top middle

r = rotate(img, 90, resize=True)
print("90 deg:")
print(r)
