import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Update compute_histogram to force ranges based on dtype
code = code.replace(
    """def compute_histogram(arr, bins=128):
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    arr_clean = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if arr_clean.size == 0:
        return {'counts': [], 'bin_edges': []}
    counts, bin_edges = np.histogram(arr_clean, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}""",
    """def compute_histogram(arr, bins=128):
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}
    arr_clean = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if arr_clean.size == 0:
        return {'counts': [], 'bin_edges': []}

    # Determine absolute range based on dtype
    range_val = None
    if arr.dtype == np.uint8:
        range_val = (0, 255)
    elif arr.dtype == np.uint16:
        range_val = (0, 65535)
    elif arr.dtype == bool:
        range_val = (0, 1)

    if range_val is not None:
        counts, bin_edges = np.histogram(arr_clean, bins=bins, range=range_val)
    else:
        counts, bin_edges = np.histogram(arr_clean, bins=bins)

    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}"""
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
