import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    content = f.read()

# Let's replace the TNIAAnnotatorWidget completely.
# First, let's locate the block for TNIAAnnotatorWidget

# Pattern for TNIAAnnotatorWidget
pattern = r"class TNIAAnnotatorWidget\(TNIASliceWidget\):.*?(?=class TNIAScatterWidget)"

import re
match = re.search(pattern, content, flags=re.DOTALL)
if match:
    print("Found TNIAAnnotatorWidget")
