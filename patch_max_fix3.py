import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

code = code.replace(
"""            # Auto-calculate 99.5 percentile if vmax is not explicitly set
            if vmax_val is None:
                if np.issubdtype(self.im_orig.dtype, np.integer) or self.im_orig.dtype == bool:
                    vmax_val = float(np.max(self.im_orig))
                else:
                    vmax_val = float(np.percentile(self.im_orig, 99.9))""",
"""            # Auto-calculate max if vmax is not explicitly set
            if vmax_val is None:
                vmax_val = float(np.max(self.im_orig))"""
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
