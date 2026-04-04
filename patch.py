import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Replace percentile in create_multichannel_rgb
code = re.sub(
    r"m_xy = float\(np\.percentile\(xy_list\[i\], 99\.5\)\)\n\s+m_xz = float\(np\.percentile\(xz_list\[i\], 99\.5\)\)\n\s+m_zy = float\(np\.percentile\(zy_list\[i\], 99\.5\)\)",
    """if np.issubdtype(xy_list[i].dtype, np.integer) or xy_list[i].dtype == bool:
                m_xy = float(np.max(xy_list[i]))
                m_xz = float(np.max(xz_list[i]))
                m_zy = float(np.max(zy_list[i]))
            else:
                m_xy = float(np.percentile(xy_list[i], 99.9))
                m_xz = float(np.percentile(xz_list[i], 99.9))
                m_zy = float(np.percentile(zy_list[i], 99.9))""",
    code, count=1
)

# Replace percentile in create_multichannel_rgb_cmap
code = re.sub(
    r"m_xy = float\(np\.percentile\(xy_list\[i\], 99\.5\)\)\n\s+m_xz = float\(np\.percentile\(xz_list\[i\], 99\.5\)\)\n\s+m_zy = float\(np\.percentile\(zy_list\[i\], 99\.5\)\)",
    """if np.issubdtype(xy_list[i].dtype, np.integer) or xy_list[i].dtype == bool:
                m_xy = float(np.max(xy_list[i]))
                m_xz = float(np.max(xz_list[i]))
                m_zy = float(np.max(zy_list[i]))
            else:
                m_xy = float(np.percentile(xy_list[i], 99.9))
                m_xz = float(np.percentile(xz_list[i], 99.9))
                m_zy = float(np.percentile(zy_list[i], 99.9))""",
    code, count=1
)

# Replace in blend_colors
code = re.sub(
    r"vmax_c = np\.nanpercentile\(arr, 99\.5\) if vmax\[c\] is None else vmax\[c\]",
    """if vmax[c] is None:
            if np.issubdtype(intensities[:, c].dtype, np.integer) or intensities[:, c].dtype == bool:
                vmax_c = np.nanmax(arr)
            else:
                vmax_c = np.nanpercentile(arr, 99.9)
        else:
            vmax_c = vmax[c]""",
    code, count=1
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
