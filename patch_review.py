import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Fix compute_histogram NaN bug
code = code.replace(
    """def compute_histogram(arr, bins=128):
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    counts, bin_edges = np.histogram(arr, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}""",
    """def compute_histogram(arr, bins=128):
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    arr_clean = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if arr_clean.size == 0:
        return {'counts': [], 'bin_edges': []}
    counts, bin_edges = np.histogram(arr_clean, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}"""
)

# Fix AttributeError in TNIASliceWidget._render where self.im_orig can be a list
code = code.replace(
    """            if np.issubdtype(self.im_orig.dtype, np.integer) or self.im_orig.dtype == bool:
                vmax_val = float(np.max(self.im_orig))
            else:
                vmax_val = float(np.percentile(self.im_orig, 99.9))""",
    """            im_arr = np.asarray(self.im_orig)
            if np.issubdtype(im_arr.dtype, np.integer) or im_arr.dtype == bool:
                vmax_val = float(np.max(im_arr))
            else:
                vmax_val = float(np.percentile(im_arr, 99.9))"""
)


with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
