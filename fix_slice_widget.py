import re
with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'r') as f:
    content = f.read()

bad = """        self._pixel_sizes_given = pixel_sizes is not None

        self.figsize = figsize
        self.colormap = colormap
        self.colors_orig = colormap"""

# The issue is that the code below this expects `colors_orig` to map to `colors` from __init__, and `colormap` to fallback.
# But in our signature `colors` is gone, we only have `colormap`.
# Let's check `self.colors_orig` usage.
