import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Replace in TNIASliceWidget._render
code = re.sub(
    r"vmax_val = float\(np\.percentile\(self\.im_orig, 99\.5\)\)",
    """if np.issubdtype(self.im_orig.dtype, np.integer) or self.im_orig.dtype == bool:
                    vmax_val = float(np.max(self.im_orig))
                else:
                    vmax_val = float(np.percentile(self.im_orig, 99.9))""",
    code, count=1
)

# Replace in TNIAScatterWidget._render
code = re.sub(
    r"vmax_c = np\.nanpercentile\(vals, 99\.5\) if vmax_resolved\[0\] is None else vmax_resolved\[0\]",
    """if vmax_resolved[0] is None:
                    if np.issubdtype(vals.dtype, np.integer) or vals.dtype == bool:
                        vmax_c = np.nanmax(vals)
                    else:
                        vmax_c = np.nanpercentile(vals, 99.9)
                else:
                    vmax_c = vmax_resolved[0]""",
    code, count=1
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
