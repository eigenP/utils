import numpy as np
import matplotlib.pyplot as plt

def test_rotate():
    y_line = 5
    rot = 30 # degrees
    w_orig = 10
    h_orig = 10

    # 0, y_line to w_orig, y_line
    x0, y0 = 0, y_line
    x1, y1 = w_orig, y_line

    # cx, cy is the center of the original image
    cx, cy = w_orig / 2, h_orig / 2

    # when we rotate the image by rot, where do the line points go?
    rad = np.radians(-rot) # image rotation is counter-clockwise, so to find point we rotate point by -rot? Wait
    # Or image rotation rotates coordinates?
    # Actually skimage.transform.rotate rotates the image counter-clockwise around its center.
    # So a point (x, y) relative to center becomes:

    # The new center cx_new, cy_new is just half of the new dimensions.
    # W_new, H_new
    c_rot, s_rot = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    w_new = int(np.ceil(abs(w_orig * c_rot) + abs(h_orig * s_rot)))
    h_new = int(np.ceil(abs(w_orig * s_rot) + abs(h_orig * c_rot)))
    cx_new, cy_new = w_new / 2, h_new / 2

    print(f"w_new: {w_new}, h_new: {h_new}")

test_rotate()
