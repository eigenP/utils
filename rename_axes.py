import re

with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'r') as f:
    content = f.read()

# Replace ax0, ax1, ax2, ax3 in show_zyx with axXY, axZY, axXZ, axBar
content = re.sub(r'\bax0\b', 'axXY', content)
content = re.sub(r'\bax1\b', 'axZY', content)
content = re.sub(r'\bax2\b', 'axXZ', content)
content = re.sub(r'\bax3\b', 'axBar', content)

# Since we replaced ax3 with axBar, we need to update the variable
# ax3_physical_width_um to axBar_physical_width_um if we want full consistency,
# or just change it to main_physical_width_um. Let's do main_physical_width_um
content = content.replace('ax3_physical_width_um', 'main_physical_width_um')

with open('src/eigenp_utils/tnia_plotting_anywidgets.py', 'w') as f:
    f.write(content)
