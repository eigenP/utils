import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Replace in create_multichannel_rgb
code = re.sub(
    r"""if np\.issubdtype\(xy_list\[i\]\.dtype, np\.integer\) or xy_list\[i\]\.dtype == bool:
                m_xy = float\(np\.max\(xy_list\[i\]\)\)
                m_xz = float\(np\.max\(xz_list\[i\]\)\)
                m_zy = float\(np\.max\(zy_list\[i\]\)\)
            else:
                m_xy = float\(np\.percentile\(xy_list\[i\], 99\.9\)\)
                m_xz = float\(np\.percentile\(xz_list\[i\], 99\.9\)\)
                m_zy = float\(np\.percentile\(zy_list\[i\], 99\.9\)\)""",
    """m_xy = float(np.max(xy_list[i]))
            m_xz = float(np.max(xz_list[i]))
            m_zy = float(np.max(zy_list[i]))""",
    code, count=1
)

# Replace in create_multichannel_rgb_cmap
code = re.sub(
    r"""if np\.issubdtype\(xy_list\[i\]\.dtype, np\.integer\) or xy_list\[i\]\.dtype == bool:
                m_xy = float\(np\.max\(xy_list\[i\]\)\)
                m_xz = float\(np\.max\(xz_list\[i\]\)\)
                m_zy = float\(np\.max\(zy_list\[i\]\)\)
            else:
                m_xy = float\(np\.percentile\(xy_list\[i\], 99\.9\)\)
                m_xz = float\(np\.percentile\(xz_list\[i\], 99\.9\)\)
                m_zy = float\(np\.percentile\(zy_list\[i\], 99\.9\)\)""",
    """m_xy = float(np.max(xy_list[i]))
            m_xz = float(np.max(xz_list[i]))
            m_zy = float(np.max(zy_list[i]))""",
    code, count=1
)

# Replace in blend_colors
code = re.sub(
    r"""if vmax\[c\] is None:
            if np\.issubdtype\(intensities\[:, c\]\.dtype, np\.integer\) or intensities\[:, c\]\.dtype == bool:
                vmax_c = np\.nanmax\(arr\)
            else:
                vmax_c = np\.nanpercentile\(arr, 99\.9\)
        else:
            vmax_c = vmax\[c\]""",
    """vmax_c = np.nanmax(arr) if vmax[c] is None else vmax[c]""",
    code, count=1
)


# Replace in TNIASliceWidget._render
code = re.sub(
    r"""im_arr = np\.asarray\(self\.im_orig\)
            if np\.issubdtype\(im_arr\.dtype, np\.integer\) or im_arr\.dtype == bool:
                vmax_val = float\(np\.max\(im_arr\)\)
            else:
                vmax_val = float\(np\.percentile\(im_arr, 99\.9\)\)""",
    """vmax_val = float(np.max(self.im_orig))""",
    code, count=1
)

# Replace in TNIAAnnotatorWidget.__init__
code = re.sub(
    r"""        def resolve_vmax\(img\):
            if np\.issubdtype\(img\.dtype, np\.integer\) or img\.dtype == bool:
                return float\(np\.max\(img\)\)
            else:
                return float\(np\.percentile\(img, 99\.9\)\)""",
    """        def resolve_vmax(img):
            return float(np.max(img))""",
    code, count=1
)

# Replace in TNIAScatterWidget._render
code = re.sub(
    r"""if vmax_resolved\[0\] is None:
                    if np\.issubdtype\(vals\.dtype, np\.integer\) or vals\.dtype == bool:
                        vmax_c = np\.nanmax\(vals\)
                    else:
                        vmax_c = np\.nanpercentile\(vals, 99\.9\)
                else:
                    vmax_c = vmax_resolved\[0\]""",
    """vmax_c = np.nanmax(vals) if vmax_resolved[0] is None else vmax_resolved[0]""",
    code, count=1
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
