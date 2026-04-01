import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I need to modify `__init__` where `self.colors_rgb` is constructed.
# original code:
#        self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]
# wait, wait. the original code is:
#        self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]

# What I will do:
#        if self.mode == 'ids' and len(self.colors_use) == 1 and is_colormap(self.colors_use[0]):
#            cmap = plt.get_cmap(self.colors_use[0])
#            self.colors_rgb = [cmap(i / max(1, self.C - 1))[:3] for i in range(self.C)]
#        else:
#            self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]

replacement = """        if self.mode == 'ids' and len(self.colors_use) == 1 and is_colormap(self.colors_use[0]):
            cmap = plt.get_cmap(self.colors_use[0])
            self.colors_rgb = [cmap(i / max(1, self.C - 1))[:3] for i in range(self.C)]
        else:
            self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]"""

code = code.replace("        self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]", replacement)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
