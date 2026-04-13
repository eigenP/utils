import re
with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'r') as f:
    content = f.read()

# Let's fix the bug. It is: `figsize=self.figsize`. `self.figsize` is somehow `['white']`.
# Wait, look at `show_zyx_max_slice_interactive`:
# `w = TNIASliceWidget(im, pixel_sizes=pixel_sizes, figsize=figsize, colormap=colormap if colormap is not None else colors...`
# If we call `show_zyx_max_slice_interactive(im, colors=['red', 'blue'])`, `colormap` is None.
# Then `colormap` becomes `['red', 'blue']`. That's passed to `TNIASliceWidget(..., colormap=['red', 'blue'])`.
# Wait, look closely at line 2176:
# `w = TNIASliceWidget(im, pixel_sizes=pixel_sizes, figsize=figsize, colormap=colormap if colormap is not None else colors,`
# But what if `colors=None` and `colormap=None`? `colormap=None`.
# Wait, the error is: `self = <[AttributeError("'Figure' object has no attribute 'bbox'") raised in repr()] Figure object...`
# `figsize = ['white']`
# Oh! `figsize` became `['white']`.
# Let's trace `TNIASliceWidget.__init__`:
# `def __init__(self, im, pixel_sizes=None, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1,...`
# Wait, let's look at how I modified `TNIASliceWidget.__init__` arguments vs `super().__init__` vs instantiation.
