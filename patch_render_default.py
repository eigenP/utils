import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# We need to insert the logic right after points are unpacked and length is known:
# e.g., after:
#             raise ValueError("points must be an array of shape (N, 3) representing (Z, Y, X) or a tuple/list of 3 arrays (Z, Y, X).")

logic = """
    if render is None:
        if len(X) < 10000:
            render = 'points'
        else:
            render = 'density'
"""

pattern = r'(raise ValueError\("points must be an array of shape \(N, 3\) representing \(Z, Y, X\) or a tuple/list of 3 arrays \(Z, Y, X\)\."\))'

code = re.sub(pattern, r'\1\n' + logic, code)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
