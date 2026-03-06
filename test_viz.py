import sys
sys.path.insert(0, 'src')
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive_point_annotator

# Large image
im = np.zeros((100, 200, 300))
w = show_xyz_max_slice_interactive_point_annotator(im)

print("min_dim for 100x200x300 is 100")
print("point_size should be max(3, ceil(0.005 * 100)) = max(3, ceil(0.5)) = 3")
print("actual point size:", w.point_size)

w.add_point(50, 50, 50)
s = w.point_size // 2

# For size 3, s=1, slice is -1 to +2 -> length 3
nz = np.count_nonzero(w._annot_img)
print("number of nonzero pixels:", nz)
expected = (s*2 + 1)**3
print("expected:", expected)
assert nz == expected
