from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
import numpy as np

img = np.zeros((10, 10, 10))
w = show_zyx_max_slice_interactive(img, slabs_thickness=(2,3,4))

print("x_t", w.x_t, "y_t", w.y_t, "z_t", w.z_t)
