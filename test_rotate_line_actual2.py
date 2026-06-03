import numpy as np

def _rotate_points_2d(px, py, angle_deg, cx, cy):
    if angle_deg == 0: return px, py
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rx = cos_t * (px - cx) - sin_t * (py - cy) + cx
    ry = sin_t * (px - cx) + cos_t * (py - cy) + cy
    return rx, ry

# The scatter plot uses `_rotate_points_2d` to rotate the points.
# Let's see what it does.
orig_w = 100
orig_h = 100
cx = orig_w / 2.0
cy = orig_h / 2.0
rx, ry = _rotate_points_2d(20, 0, 30, cx, cy)
print(f"0 -> {rx}, {ry}")
rx2, ry2 = _rotate_points_2d(20, 100, 30, cx, cy)
print(f"100 -> {rx2}, {ry2}")
