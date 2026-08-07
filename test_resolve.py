import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from eigenp_utils.tnia_plotting_anywidgets import is_colormap

def resolve_color(c):
    if isinstance(c, mcolors.Colormap):
        raise TypeError(
            f"Expected a registered colormap name or color as a string (e.g., 'viridis' or 'red'), "
            f"but got a Colormap instance: {c}. Please pass the string name of the registered colormap."
        )

    if not isinstance(c, str):
        return c

    # First check if it's a registered colormap name
    if is_colormap(c):
        cmap = plt.get_cmap(c)
        return mcolors.to_hex(cmap(1.0)[:3])  # Get hex of final color

    try:
        # Check if it's already a valid color name or hex
        mcolors.to_rgb(c)
        return c
    except ValueError:
        pass

    return c

colors_to_test = ['pink', 'gray', 'hot', 'autumn', 'spring', 'viridis']
for c in colors_to_test:
    print(f"resolve_color('{c}') -> {resolve_color(c)}")
