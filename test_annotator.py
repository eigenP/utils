import numpy as np
import traitlets
import warnings
from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget

# 6. Unbounded Point Deletion in TNIAAnnotatorWidget
try:
    im = np.zeros((10, 20, 30))
    w = TNIAAnnotatorWidget(im)
    w.points = [[5, 10, 15]]
    w.annotation_mode = True
    w.annotation_action = 'delete'
    w.axis_bounds = {'xy': (0, 0, 1, 1)}

    # Click at 0, 0 in XY plane which corresponds to data coords x=0, y=19
    # The existing point is at x=15, y=10. The squared distance is (15-0)^2 + (10-19)^2 = 225 + 81 = 306.

    w._handle_click({'new': {'plane': 'xy', 'x': 0.0, 'y': 0.0}}) # y=0 from top -> mpl y_frac=1.0 -> fraction_from_top=0.0 -> data_y=0

    print(f"Points after click: {w.points}")
except Exception as e:
    print(f"Error: {e}")
