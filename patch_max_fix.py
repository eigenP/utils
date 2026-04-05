import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I see my regex for TNIASliceWidget missed it, probably because it got reverted or wasn't matched properly. Let me fix the remaining percentiles.
code = re.sub(
    r"""            # Auto-calculate 99\.5 percentile if vmax is not explicitly set\n\s+if vmax_val is None:\n\s+vmax_val = float\(np\.percentile\(self\.im_orig, 99\.9\)\)""",
    """            # Auto-calculate np.max if vmax is not explicitly set
            if vmax_val is None:
                vmax_val = float(np.max(self.im_orig))""",
    code, count=1
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
