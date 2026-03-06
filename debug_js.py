with open("src/eigenp_utils/tnia_plotting_anywidgets.js", "r") as f:
    js = f.read()

# Let's inspect the JS click handler
import re
match = re.search(r'img\.addEventListener\("click", \(e\) => \{.*?\n        \}\);', js, flags=re.DOTALL)
if match:
    print(match.group(0))
else:
    print("Not found")
