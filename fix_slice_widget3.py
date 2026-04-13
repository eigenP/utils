import re
with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'r') as f:
    content = f.read()

# restore original fallback:
bad1 = """            # Resolve default colors to ensure stability when toggling
            if colormap is None:
                 defaults = ['lime', 'magenta', 'yellow', 'cyan', 'red', 'blue'] # We usually use 'lime' and 'magenta' for 2 channels
                 self.colors_resolved = defaults[:self.num_channels]
            elif isinstance(colormap, (list, tuple)):
                 self.colors_resolved = list(colormap)
            else:
                 self.colors_resolved = [colormap] * self.num_channels"""

good1 = """            # Resolve default colors to ensure stability when toggling
            if colormap is None:
                 defaults = ['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']
                 # Extend if needed
                 while len(defaults) < self.num_channels:
                     defaults += defaults
                 self.colors_resolved = defaults[:self.num_channels]
            else:
                 self.colors_resolved = list(colormap)"""

content = content.replace(bad1, good1)
with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'w') as f:
    f.write(content)
