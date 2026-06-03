import numpy as np
import matplotlib.pyplot as plt

def get_rotated_line(x0, y0, x1, y1, angle_deg, orig_w, orig_h, new_w, new_h):
    # In an image array where row=Y and col=X, rotate(img, angle, resize=True)
    # does counter-clockwise rotation of the image.

    # Original center
    cx_orig = (orig_w - 1) / 2.0
    cy_orig = (orig_h - 1) / 2.0

    # New center
    cx_new = (new_w - 1) / 2.0
    cy_new = (new_h - 1) / 2.0

    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)

    # Coordinate transformation for point (x, y)
    def trans(x, y):
        dx = x - cx_orig
        dy = y - cy_orig
        # Counter-clockwise rotation of points in a y-down system
        nx = dx * c + dy * s
        ny = -dx * s + dy * c
        return cx_new + nx, cy_new + ny

    return trans(x0, y0), trans(x1, y1)

# From actual rotate test
# min y: 40, x=18
# max y: 125, x=67

# orig shape: 100x100.
# new shape:
rad = np.radians(30)
new_h = int(np.ceil(100 * np.cos(rad) + 100 * np.sin(rad)))
new_w = int(np.ceil(100 * np.sin(rad) + 100 * np.cos(rad)))
print(f"new_w={new_w}, new_h={new_h}")

(x0, y0), (x1, y1) = get_rotated_line(20, 0, 20, 99, 30, 100, 100, new_w, new_h)
print(f"Pred: ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f})")
