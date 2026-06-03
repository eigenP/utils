import numpy as np
from skimage.transform import rotate

img = np.zeros((11, 11))
img[2:9, 5] = 1 # vertical line at x=5, from y=2 to y=8

img_rot = rotate(img, 30, resize=True)
y_idx, x_idx = np.where(img_rot > 0.5)

print(f"Actual points in rotated image (vertical line):")
for x, y in zip(x_idx, y_idx):
    print(f"({x}, {y})")
