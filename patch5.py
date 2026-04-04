import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    code = f.read()

# For TNIASliceWidget
code = re.sub(
    r"        if x_s is not None: self.x_s = int\(x_s\)\n\s+if y_s is not None: self.y_s = int\(y_s\)\n\s+if z_s is not None: self.z_s = int\(z_s\)\n\n\s+# Initialize observers and render\n\s+self\._init_observers\(\)",
    """        if x_s is not None: self.x_s = int(x_s)
        if y_s is not None: self.y_s = int(y_s)
        if z_s is not None: self.z_s = int(z_s)

        # Compute histograms
        hists = []
        if isinstance(im, list):
            for img in im:
                hists.append(compute_histogram(img))
        else:
            hists.append(compute_histogram(im))
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()""",
    code, count=1
)


# For TNIAScatterWidget
code = re.sub(
    r"        if z_s is None: self.z_s = int\(\(self.zmax - self.zmin\)/2\)\n\s+else: self.z_s = int\(z_s\)\n\n\s+# Initialize observers and render\n\s+self\._init_observers\(\)",
    """        if z_s is None: self.z_s = int((self.zmax - self.zmin)/2)
        else: self.z_s = int(z_s)

        # Compute histograms
        hists = []
        if self.mode == 'single' or self.mode == 'ids' or self.mode == 'idx_lists':
            # For scatter with a single set of points and uniform color, the distribution is just counts (not really intensities).
            # We don't have multiple intensity channels here. We'll leave it empty.
            # If ids, we just have 1 channel (the IDs).
            if self.mode == 'ids':
                hists.append(compute_histogram(self.ch_ids))
            else:
                hists.append({'counts': [], 'bin_edges': []}) # No intensities to map
        elif self.mode == 'cont_single':
            hists.append(compute_histogram(self.cont_single))
        elif self.mode == 'cont_multi':
            for c in range(self.C):
                hists.append(compute_histogram(self.cont_multi[:, c]))

        # pad to C
        while len(hists) < self.C:
            hists.append({'counts': [], 'bin_edges': []})
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()""",
    code, count=1
)


with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(code)
