from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget, TNIASliceWidget
import numpy as np

# Original code snippet in TNIAAnnotatorWidget.__init__
# def __init__(self, *args, **kwargs):
#    super().__init__(*args, **kwargs)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    content = f.read()

import re

# We will modify TNIAAnnotatorWidget.__init__
# Also we will modify the factory function show_xyz_max_slice_interactive_point_annotator
