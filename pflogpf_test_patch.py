import re

with open("tests/test_single_cell.py", "r") as f:
    content = f.read()

# Add pflogpf to the import list from single_cell
if "from eigenp_utils.single_cell import (" in content:
    content = content.replace("from eigenp_utils.single_cell import (", "from eigenp_utils.single_cell import (\n    pflogpf,")
else:
    # try to append to the end of existing single_cell imports
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("from eigenp_utils.single_cell import "):
            lines[i] = line + ", pflogpf"
            break
    else:
        # Just insert it after imports
        content = "from eigenp_utils.single_cell import pflogpf\n" + content

if type(content) == list:
    content = "\n".join(lines)

with open("tests/test_single_cell.py", "w") as f:
    f.write(content)
