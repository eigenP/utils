import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Add histograms traitlet to TNIAWidgetBase
code = code.replace(
    "    opacity_list = traitlets.List(traitlets.Float()).tag(sync=True)",
    "    opacity_list = traitlets.List(traitlets.Float()).tag(sync=True)\n    histograms_data = traitlets.List(traitlets.Dict()).tag(sync=True)"
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
