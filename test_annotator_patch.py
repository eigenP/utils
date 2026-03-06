import sys
sys.path.insert(0, 'src')
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive_point_annotator

im = np.zeros((10, 20, 30))
w = show_xyz_max_slice_interactive_point_annotator(im)

print("num_channels:", w.num_channels)
print("channel_names:", w.channel_names)
print("colors_resolved:", w.colors_resolved)
print("channel_visible:", w.channel_visible)

w.annotation_mode = True
w.points = [[15, 10, 5]]
w._render_wrapper()

print("Did re-render work?")
