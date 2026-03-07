import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive_point_annotator

im = np.zeros((10, 20, 30))
w = show_xyz_max_slice_interactive_point_annotator(im)
