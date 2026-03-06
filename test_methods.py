import sys
sys.path.insert(0, 'src')
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive_point_annotator

im = np.zeros((10, 20, 30))
w = show_xyz_max_slice_interactive_point_annotator(im)

print("Points before:", w.points)
w.add_point(15, 10, 5)
print("Points after add:", w.points)

# The observer `_on_points_changed` should have updated `_annot_img`
print("Annot image max:", w._annot_img.max())

w.remove_point(15, 10, 5)
print("Points after remove:", w.points)
print("Annot image max after remove:", w._annot_img.max())
