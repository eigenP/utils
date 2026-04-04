import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# Update TNIAAnnotatorWidget.__init__ to use 99.9 percentile for default vmax
code = code.replace(
    """        # Ensure vmax is padded correctly for the annotation channel
        vmax = kwargs.get('vmax', None)
        if vmax is None:
            vmax_list = [None] * (len(im_list) - 1) + [255.0]
            kwargs['vmax'] = vmax_list
        elif isinstance(vmax, (list, tuple)):
            vmax_list = list(vmax)
            while len(vmax_list) < len(im_list) - 1:
                vmax_list.append(None)
            vmax_list = vmax_list[:len(im_list) - 1]
            vmax_list.append(255.0)
            kwargs['vmax'] = vmax_list
        else:
            vmax_list = [vmax] * (len(im_list) - 1) + [255.0]
            kwargs['vmax'] = vmax_list""",
    """        # Ensure vmax is padded correctly for the annotation channel
        vmax = kwargs.get('vmax', None)

        def resolve_vmax(img):
            if np.issubdtype(img.dtype, np.integer) or img.dtype == bool:
                return float(np.max(img))
            else:
                return float(np.percentile(img, 99.9))

        if vmax is None:
            vmax_list = [resolve_vmax(im_list[i]) for i in range(len(im_list) - 1)] + [255.0]
            kwargs['vmax'] = vmax_list
        elif isinstance(vmax, (list, tuple)):
            vmax_list = list(vmax)
            while len(vmax_list) < len(im_list) - 1:
                vmax_list.append(None)
            vmax_list = vmax_list[:len(im_list) - 1]
            for i in range(len(vmax_list)):
                if vmax_list[i] is None:
                    vmax_list[i] = resolve_vmax(im_list[i])
            vmax_list.append(255.0)
            kwargs['vmax'] = vmax_list
        else:
            vmax_list = [vmax] * (len(im_list) - 1) + [255.0]
            kwargs['vmax'] = vmax_list"""
)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
