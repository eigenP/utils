from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
import numpy as np

img = np.zeros((10, 10, 10))
w = show_zyx_max_slice_interactive(img)

print(w.trait_names())
