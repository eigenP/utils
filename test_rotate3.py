import numpy as np

theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)

cx, cy = 5, 5
cx_new, cy_new = 7, 7

dx = 2 - cx
dy = 5 - cy

# skimage rotates the image counter-clockwise. In an array where y goes down,
# counter-clockwise means x moves to +y, y moves to -x.
# x' = x*cos + y*sin
# y' = -x*sin + y*cos
new_dx = dx * c + dy * s
new_dy = -dx * s + dy * c
pred_x = cx_new + new_dx
pred_y = cy_new + new_dy
print(f"Pred: ({pred_x:.1f}, {pred_y:.1f})")

dx2 = 8 - cx
new_dx2 = dx2 * c + dy * s
new_dy2 = -dx2 * s + dy * c
pred_x2 = cx_new + new_dx2
pred_y2 = cy_new + new_dy2
print(f"Pred2: ({pred_x2:.1f}, {pred_y2:.1f})")
