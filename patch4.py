import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I will add a method to compute histograms in TNIAWidgetBase and call it from derived classes.
# But actually, the data is different for slice vs scatter widgets.
# I'll just add the helper function and compute them in `__init__` for TNIASliceWidget and TNIAScatterWidget.
code = code.replace(
    "class TNIAWidgetBase(anywidget.AnyWidget):",
    """def compute_histogram(arr, bins=128):
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    counts, bin_edges = np.histogram(arr, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}

class TNIAWidgetBase(anywidget.AnyWidget):"""
)


with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
