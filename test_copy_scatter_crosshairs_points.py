from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
import numpy as np

N = 10
X = np.random.rand(N) * 10
Y = np.random.rand(N) * 10
Z = np.random.rand(N) * 10
w = show_zyx_max_scatter_interactive((Z, Y, X), slabs_thickness=(2,3,4), rotate_view=(30, 40, 50))

print("x_t", w.x_t, "y_t", w.y_t, "z_t", w.z_t)
