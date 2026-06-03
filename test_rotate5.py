import numpy as np

theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)

cx, cy = 5, 5
cx_new, cy_new = 7, 7

orig_y1, orig_y2 = 2, 8
orig_x = 5

for orig_y in [orig_y1, orig_y2]:
    dx = orig_x - cx
    dy = orig_y - cy
    new_dx = dx * c + dy * s
    new_dy = -dx * s + dy * c
    pred_x = cx_new + new_dx
    pred_y = cy_new + new_dy
    print(f"Orig: ({orig_x}, {orig_y}) -> Pred: ({pred_x:.1f}, {pred_y:.1f})")
