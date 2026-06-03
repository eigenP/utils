import numpy as np
import matplotlib.pyplot as plt

# Center computation is identical to the one in `rotate_view` behavior.
# It rotates the bounds.

x_lims = [20, 80]
y_lims = [10, 90]
z_lims = [30, 70]
sx = 1.0
sy = 1.0
sz = 1.0

# When an image is rotated by rotate(img, angle, resize=True),
# the original center of the image (w/2, h/2) is moved to the center
# of the new rotated image (w_new/2, h_new/2).

def plot_rotated_lines(orig_w, orig_h, rot_deg, lines):
    # center of original image
    cx_orig = orig_w / 2.0
    cy_orig = orig_h / 2.0

    # size of new image
    rad = np.radians(rot_deg)
    c, s = np.abs(np.cos(rad)), np.abs(np.sin(rad))
    w_new = int(np.ceil(orig_w * c + orig_h * s))
    h_new = int(np.ceil(orig_w * s + orig_h * c))

    cx_new = w_new / 2.0
    cy_new = h_new / 2.0

    # rotate angle is counter-clockwise?
    # actually skimage rotates the image, meaning points move.
    # a point relative to center: dx = x - cx, dy = y - cy
    # x' = cx_new + dx * cos(a) + dy * sin(a)
    # y' = cy_new - dx * sin(a) + dy * cos(a)  <- verify standard vs y-down

    c_ang = np.cos(np.radians(rot_deg))
    s_ang = np.sin(np.radians(rot_deg))

    # We want to plot a line from (x0, y0) to (x1, y1)

    res = []
    for line in lines:
        pts = []
        for (x, y) in line:
            dx = x - cx_orig
            dy = y - cy_orig

            # The rotation in skimage is counter-clockwise.
            # In a (column, row) coordinate system where row (y) goes down,
            # counter-clockwise rotation means:
            # dx' = dx * c + dy * s
            # dy' = -dx * s + dy * c
            # NOTE: this assumes center is at cx_orig, cy_orig.
            # However, skimage does:
            # coordinate transformation from target to source.

            x_new = cx_new + dx * c_ang + dy * s_ang
            y_new = cy_new - dx * s_ang + dy * c_ang
            pts.append((x_new, y_new))

        res.append(pts)

    return res

lines = [
    # vertical line at x=x_lims[0], from y=0 to y=100
    [(x_lims[0], 0), (x_lims[0], 100)]
]

res = plot_rotated_lines(100, 100, 30, lines)
print(res)
