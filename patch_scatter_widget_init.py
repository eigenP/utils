import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I need to change how `super().__init__` is called and remove the assignments
# of vmin, vmax, gamma, opacity. However, wait, TNIAWidgetBase does not take `vmin`, `vmax`, `gamma`, `opacity` in `__init__`. Let's check `TNIAWidgetBase.__init__`.
