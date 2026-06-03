import numpy as np
from skimage.transform import rotate
import matplotlib.pyplot as plt

img = np.zeros((11, 11))
img[5, 2:9] = 1 # horizontal line at y=5, from x=2 to x=8

img_rot = rotate(img, 30, resize=True)
print(f"orig shape: {img.shape}, rot shape: {img_rot.shape}")

# Find where the line is now
y_idx, x_idx = np.where(img_rot > 0.5)

# Center of original
cx, cy = 5, 5
# Center of new
cx_new, cy_new = img_rot.shape[1]/2 - 0.5, img_rot.shape[0]/2 - 0.5

print(f"cx_new={cx_new}, cy_new={cy_new}")

# We want to rotate the line coordinates (x, y) around (cx, cy)
# skimage rotates counter-clockwise.
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)

for orig_x in [2, 8]:
    orig_y = 5

    dx = orig_x - cx
    dy = orig_y - cy

    # skimage rotates the *image* counter-clockwise.
    # This means the *coordinates* of the points move counter-clockwise.
    # Note: in image coords, y points down.
    # Counter-clockwise rotation in a y-down system means:
    # new_x = dx * cos - dy * sin
    # new_y = dx * sin + dy * cos  <-- but wait, standard math is y-up.
    # Let's test standard math rotation:
    new_dx = dx * c + dy * s
    new_dy = -dx * s + dy * c

    # skimage rotate docs say: "rotation angle in degrees in counter-clockwise direction."

    pred_x = cx_new + new_dx
    pred_y = cy_new + new_dy
    print(f"Orig: ({orig_x}, {orig_y}) -> Pred: ({pred_x:.1f}, {pred_y:.1f})")

print(f"Actual points in rotated image:")
for x, y in zip(x_idx, y_idx):
    print(f"({x}, {y})")
