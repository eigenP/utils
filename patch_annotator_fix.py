import re

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "r") as f:
    content = f.read()

replacement = """class TNIAAnnotatorWidget(TNIASliceWidget):
    \"\"\"
    Subclass of TNIASliceWidget that supports interactive point annotation.
    \"\"\"
    # UI Toggles
    annotation_mode = traitlets.Bool(False).tag(sync=True)
    annotation_action = traitlets.Unicode('add').tag(sync=True) # 'add' or 'delete'

    # Communication
    click_coords = traitlets.Dict().tag(sync=True) # {'plane': 'xy', 'x': 0.5, 'y': 0.5, 't': 123}

    # Data
    points = traitlets.List().tag(sync=True) # List of [x, y, z] lists
    axis_bounds = traitlets.Dict().tag(sync=True) # Bounding boxes of axes in figure coords

    def __init__(self, im, colors=None, *args, **kwargs):
        # Normalize input to list
        if not isinstance(im, list):
            im_list = [im]
        else:
            im_list = list(im)

        # Ensure colors is a list
        if colors is None:
            colors_list = ['magenta', 'cyan', 'yellow', 'green', 'blue', 'orange']
        elif isinstance(colors, str):
            colors_list = [colors]
        else:
            colors_list = list(colors)

        while len(colors_list) < len(im_list):
            colors_list.extend(['magenta', 'cyan', 'yellow', 'green', 'blue', 'orange'])

        colors_list = colors_list[:len(im_list)]

        # Get shape
        im_shape = im_list[0].shape
        Z, Y, X = im_shape
        min_dim = min(X, Y, Z)
        self.point_size = max(3, int(np.ceil(0.005 * min_dim)))

        # Create persistent annotation channel
        self._annot_img = np.zeros((Z, Y, X), dtype=np.uint8)

        # Append annotation channel
        im_list.append(self._annot_img)
        colors_list.append('red')

        # Initialize superclass
        super().__init__(im_list, colors=colors_list, *args, **kwargs)

        # Override the last channel name
        self.channel_names = self.channel_names[:-1] + ['Annotations']

    def _init_observers(self):
        super()._init_observers()
        self.observe(self._handle_click, names=['click_coords'])
        self.observe(self._on_points_changed, names=['points'])
        # Also re-render if annotation_mode changes (so UI cursor updates)
        self.observe(self._render_wrapper, names=['annotation_mode'])

    def _handle_click(self, change):
        if not self.annotation_mode:
            return

        coords = change.new if hasattr(change, 'new') else change.get('new', {})
        if not coords:
            return

        plane = coords.get('plane')
        frac_x = coords.get('x')
        frac_y = coords.get('y')

        bounds = self.axis_bounds.get(plane)
        if not bounds:
            return

        b_x0, b_y0, b_w, b_h = bounds

        # Check if click is inside this axis
        # Note: JS y_frac is from top-left. Matplotlib bounds are from bottom-left.
        mpl_y_frac = 1.0 - frac_y

        if not (b_x0 <= frac_x <= b_x0 + b_w and b_y0 <= mpl_y_frac <= b_y0 + b_h):
            return

        local_x = (frac_x - b_x0) / b_w
        local_y_mpl = (mpl_y_frac - b_y0) / b_h
        fraction_from_top = 1.0 - local_y_mpl

        x0 = max(0, self.x_s - self.x_t)
        x1 = min(self.dims[2] - 1, self.x_s + self.x_t)
        y0 = max(0, self.y_s - self.y_t)
        y1 = min(self.dims[1] - 1, self.y_s + self.y_t)
        z0 = max(0, self.z_s - self.z_t)
        z1 = min(self.dims[0] - 1, self.z_s + self.z_t)

        if plane == 'xy':
            data_x = int(local_x * self.dims[2])
            data_y = int(fraction_from_top * self.dims[1])
            data_z = self.z_s
        elif plane == 'zy':
            data_z = int(local_x * self.dims[0])
            data_y = int(fraction_from_top * self.dims[1])
            data_x = self.x_s
        elif plane == 'xz':
            data_x = int(local_x * self.dims[2])
            data_z = int(fraction_from_top * self.dims[0])
            data_y = self.y_s
        else:
            return

        data_x = max(0, min(self.dims[2] - 1, data_x))
        data_y = max(0, min(self.dims[1] - 1, data_y))
        data_z = max(0, min(self.dims[0] - 1, data_z))

        if self.annotation_action == 'add':
            self.add_point(data_x, data_y, data_z)
        elif self.annotation_action == 'delete':
            if not self.points: return

            pts = np.array(self.points)
            if plane == 'xy':
                mask = (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 0] - data_x)**2 + (visible_pts[:, 1] - data_y)**2
            elif plane == 'zy':
                mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 2] - data_z)**2 + (visible_pts[:, 1] - data_y)**2
            elif plane == 'xz':
                mask = (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
                if not np.any(mask): return
                visible_pts = pts[mask]
                dist = (visible_pts[:, 0] - data_x)**2 + (visible_pts[:, 2] - data_z)**2

            closest_idx_in_visible = np.argmin(dist)
            closest_pt = visible_pts[closest_idx_in_visible]
            self.remove_point(closest_pt[0], closest_pt[1], closest_pt[2])

    def add_point(self, x, y, z):
        \"\"\"Programmatically add a point\"\"\"
        pt = [int(x), int(y), int(z)]
        if pt not in self.points:
            self.points = self.points + [pt]

    def remove_point(self, x, y, z):
        \"\"\"Programmatically remove a point\"\"\"
        pt = [int(x), int(y), int(z)]
        new_points = []
        deleted = False
        for p in self.points:
            if not deleted and p == pt:
                deleted = True
                continue
            new_points.append(p)
        if deleted:
            self.points = new_points

    def _on_points_changed(self, change):
        # Update annotation mask efficiently
        self._annot_img.fill(0)
        Z, Y, X = self.dims
        s = self.point_size // 2
        for p in self.points:
            px, py, pz = p
            z0 = max(0, pz - s)
            z1 = min(Z, pz + s + 1)
            y0 = max(0, py - s)
            y1 = min(Y, py + s + 1)
            x0 = max(0, px - s)
            x1 = min(X, px + s + 1)
            self._annot_img[z0:z1, y0:y1, x0:x1] = 255

        self._render_wrapper(change)

    def _render_wrapper(self, change=None):
        fig = super()._render()
        if fig:
            if len(fig.axes) >= 3:
                ax_xy = fig.axes[0]
                ax_zy = fig.axes[1]
                ax_xz = fig.axes[2]

                def get_bounds(ax):
                    bbox = ax.get_position()
                    return [bbox.x0, bbox.y0, bbox.width, bbox.height]

                self.axis_bounds = {
                    'xy': get_bounds(ax_xy),
                    'zy': get_bounds(ax_zy),
                    'xz': get_bounds(ax_xz)
                }

            buf = io.BytesIO()
            fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
            fig.savefig(buf, format='png')

            if len(fig.axes) >= 3:
                self.axis_bounds = {
                    'xy': get_bounds(fig.axes[0]),
                    'zy': get_bounds(fig.axes[1]),
                    'xz': get_bounds(fig.axes[2])
                }

            self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
"""

pattern = r"class TNIAAnnotatorWidget\(TNIASliceWidget\):.*?(?=class TNIAScatterWidget)"
content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("src/eigenp_utils/tnia_plotting_anywidgets.py", "w") as f:
    f.write(content)
