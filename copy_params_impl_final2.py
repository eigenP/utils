import numpy as np

def _get_rotated_line_scatter(x0, y0, x1, y1, angle_deg, orig_w, orig_h, cx_data, cy_data, sx, sy):
    # Scatter does NOT use skimage rotate. It rotates the coordinates around cx_data, cy_data directly.
    if angle_deg == 0:
        return [x0*sx, x1*sx], [y0*sy, y1*sy]
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    def trans(px, py):
        # inputs are data coordinates
        # cx_data is data center
        rx = cos_t * (px*sx - cx_data*sx) - sin_t * (py*sy - cy_data*sy) + cx_data*sx
        ry = sin_t * (px*sx - cx_data*sx) + cos_t * (py*sy - cy_data*sy) + cy_data*sy
        return rx, ry

    rx0, ry0 = trans(x0, y0)
    rx1, ry1 = trans(x1, y1)
    return [rx0, rx1], [ry0, ry1]
