import numpy as np

def _get_rotated_line(x0, y0, x1, y1, angle_deg, orig_w, orig_h):
    if angle_deg == 0.0:
        return [x0, x1], [y0, y1]

    rad = np.radians(angle_deg)
    c, s = np.cos(rad), np.sin(rad)

    # Calculate new dimensions just like skimage.transform.rotate(..., resize=True)
    new_w = int(np.ceil(abs(orig_w * c) + abs(orig_h * s)))
    new_h = int(np.ceil(abs(orig_w * s) + abs(orig_h * c)))

    cx_orig = (orig_w - 1) / 2.0
    cy_orig = (orig_h - 1) / 2.0
    cx_new = (new_w - 1) / 2.0
    cy_new = (new_h - 1) / 2.0

    def trans(x, y):
        dx = x - cx_orig
        dy = y - cy_orig
        nx = dx * c + dy * s
        ny = -dx * s + dy * c
        return cx_new + nx, cy_new + ny

    rx0, ry0 = trans(x0, y0)
    rx1, ry1 = trans(x1, y1)
    return [rx0, rx1], [ry0, ry1]
