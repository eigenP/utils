import sys
sys.path.insert(0, 'src')
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive_point_annotator
from traitlets import Bunch

im = np.zeros((10, 20, 30))
w = show_xyz_max_slice_interactive_point_annotator(im)
w.annotation_mode = True

w._render_wrapper()

xy_bounds = w.axis_bounds['xy']
bx0, by0, bw, bh = xy_bounds

# Simulate click from top-left logic (JS behavior)
# we click in the middle of the xy plot
frac_x = bx0 + bw/2

# if y is in the middle of the plot from bottom, then mpl_y_frac = by0 + bh/2
mpl_y_frac = by0 + bh/2

# but JS sends fraction from top:
# mpl_y_frac = 1.0 - frac_y => frac_y = 1.0 - mpl_y_frac
frac_y = 1.0 - mpl_y_frac

print("simulating click at", frac_x, frac_y)
w._handle_click(Bunch(new={'plane': 'xy', 'x': frac_x, 'y': frac_y}))
print("points:", w.points)
