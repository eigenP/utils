import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

code = code.replace(
"""            # Auto-calculate 99.5 percentile if vmax is not explicitly set
            if vmax_val is None:
                im_arr = np.asarray(self.im_orig)
                if np.issubdtype(im_arr.dtype, np.integer) or im_arr.dtype == bool:
                    vmax_val = float(np.max(im_arr))
                else:
                    vmax_val = float(np.percentile(im_arr, 99.9))""",
"""            # Auto-calculate np.max if vmax is not explicitly set
            if vmax_val is None:
                vmax_val = float(np.max(np.asarray(self.im_orig)))"""
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
