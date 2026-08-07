import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eigenp_utils.tnia_plotting_anywidgets import IsoScatterWidget, TNIASliceWidget, TNIAScatterWidget, resolve_color, blend_colors, show_zyx_max_scatter_interactive, TNIAAnnotatorWidget

# 1. IsoScatterWidget Instantiation Crash
try:
    X, Y, Z = np.array([]), np.array([]), np.array([])
    widget = IsoScatterWidget(X, Y, Z)
    print("IsoScatterWidget instantiated successfully for empty data")
except Exception as e:
    print(f"IsoScatterWidget instantiation failed: {e}")

try:
    X, Y, Z = np.random.rand(10), np.random.rand(10), np.random.rand(10)
    widget = IsoScatterWidget(X, Y, Z)
    print("IsoScatterWidget instantiated successfully for non-empty data")
except Exception as e:
    print(f"IsoScatterWidget instantiation failed: {e}")

# 3. Colormap/Color Name Collision in resolve_color
colors_to_test = ['pink', 'gray', 'hot', 'autumn', 'spring', 'viridis']
for c in colors_to_test:
    print(f"resolve_color('{c}') -> {resolve_color(c)}")

# 4. blend_colors Auto-Range Overwritten
intensities = np.array([[0.5, 2.0], [1.5, 4.0], [np.nan, 3.0]])
base_colors = ['red', 'blue']
try:
    blended = blend_colors(intensities, base_colors, vmin=[None, None], vmax=[None, None])
    print(f"blend_colors completed, vmin handling: success")
except Exception as e:
    print(f"blend_colors failed: {e}")
