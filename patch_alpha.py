import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I will revert the alpha hack and do it correctly.
code = code.replace("alpha=opacity_resolved[0] if len(opacity_resolved) == 1 else opacity_resolved[c] if 'c' in locals() else opacity_resolved[0]", "alpha=self.alpha")

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
