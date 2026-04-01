import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# I need to implement the list resolving inside TNIAScatterWidget.__init__ right after `self.colors_rgb` is set,
# similar to what TNIASliceWidget does. Then I'll replace `self.vmin`, `self.vmax`, `self.gamma`, `self.opacity` usages inside `_render`.

# First, let's locate `self.vmin = vmin` assignments and remove them.
code = re.sub(r'        self\.vmin = vmin\n', '', code)
code = re.sub(r'        self\.vmax = vmax\n', '', code)
code = re.sub(r'        self\.gamma = gamma\n', '', code)
code = re.sub(r'        self\.opacity = opacity\n', '', code)

# Let's locate the place where we can initialize the lists. We'll do this after setting `self.C` and `self.colors_rgb`.
# Right before `# Init values\n        if x_t is not None:`
init_lists_logic = """
        # Set traitlets lists for interactive parameters
        def _to_list(val, n, default):
            if val is None:
                return [default] * n
            elif isinstance(val, (list, tuple)):
                if len(val) >= n:
                    return list(val[:n])
                else:
                    return list(val) + [default] * (n - len(val))
            else:
                return [val] * n

        def _resolve_vmin_vmax(val, n):
            lst = _to_list(val, n, None)
            return ["" if x is None else x for x in lst]

        self.vmin_list = _resolve_vmin_vmax(vmin, self.C)
        self.vmax_list = _resolve_vmin_vmax(vmax, self.C)
        self.gamma_list = _to_list(gamma, self.C, 1.0)
        self.opacity_list = _to_list(opacity, self.C, 1.0)

        self.channel_names = [f"Channel {i}" for i in range(self.C)]
        self.channel_dtypes = ["float"] * self.C
        self.channel_colors = [matplotlib.colors.to_hex(c) for c in self.colors_rgb]
"""

code = code.replace(
    "        # Init values\n        if x_t is not None: self.x_t = int(x_t)",
    init_lists_logic + "\n        # Init values\n        if x_t is not None: self.x_t = int(x_t)"
)

# Replace `self.vmin`, `self.vmax`, `self.gamma`, `self.opacity` in `_render` with `vmin_val`, `vmax_val`, `gamma_val`, `opacity_val` where appropriate.
# Inside `_render`:
# We should probably define `vmin_resolved`, `vmax_resolved`, `gamma_resolved`, `opacity_resolved` lists.
code_render_start = """    def _render(self):
        # Translate widget relative coordinates (0..Dim) to data coordinates (min..max)"""
code_render_replacement = """    def _render(self):
        vmin_resolved = [None if v == "" else float(v) for v in self.vmin_list]
        vmax_resolved = [None if v == "" else float(v) for v in self.vmax_list]
        gamma_resolved = [float(g) for g in self.gamma_list]
        opacity_resolved = [float(o) for o in self.opacity_list]

        # Translate widget relative coordinates (0..Dim) to data coordinates (min..max)"""
code = code.replace(code_render_start, code_render_replacement)

# Replace `self.vmin` with `vmin_resolved[0]` or just pass the full list to create_multichannel_rgb.
# Wait, `create_multichannel_rgb` expects single vmin/vmax if it's a single value, or lists. Since it accepts lists, we can just pass the lists!
# Wait, let's check `create_multichannel_rgb` signature.
code = code.replace("vmin=self.vmin", "vmin=vmin_resolved")
code = code.replace("vmax=self.vmax", "vmax=vmax_resolved")
code = code.replace("gamma=self.gamma", "gamma=gamma_resolved")
code = code.replace("opacity=self.opacity", "opacity=opacity_resolved")
code = code.replace("alpha=self.alpha", "alpha=opacity_resolved[0] if len(opacity_resolved) == 1 else opacity_resolved[c] if 'c' in locals() else opacity_resolved[0]") # Wait, this could break. Let's fix this more carefully.
with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
