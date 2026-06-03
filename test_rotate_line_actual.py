import numpy as np
from skimage.transform import rotate

img = np.zeros((100, 100))
img[:, 20] = 1 # vertical line at x=20

img_rot = rotate(img, 30, resize=True)
y_idx, x_idx = np.where(img_rot > 0.5)

print(f"min y: {np.min(y_idx)}, max y: {np.max(y_idx)}")
print(f"at min y, x is: {x_idx[np.argmin(y_idx)]}")
print(f"at max y, x is: {x_idx[np.argmax(y_idx)]}")
