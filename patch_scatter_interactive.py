import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Modify `show_zyx_max_scatter_interactive`
# Original definition:
# def show_zyx_max_scatter_interactive(
#     points,
#     channels=None,
#     sxy=None, sz=None,
#     render='density',
#     bins=512,

# New definition
# def show_zyx_max_scatter_interactive(
#     points,
#     channels=None,
#     sxy=None, sz=None,
#     render=None,
#     bins=512,

code = re.sub(
    r"def show_zyx_max_scatter_interactive\(\s*points,\s*channels=None,\s*sxy=None,\s*sz=None,\s*render='density',",
    r"def show_zyx_max_scatter_interactive(\n    points,\n    channels=None,\n    sxy=None, sz=None,\n    render=None,",
    code
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
