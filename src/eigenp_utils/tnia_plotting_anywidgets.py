# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anywidget",
#     "traitlets",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
# ]
# ///

import pathlib
import anywidget
import traitlets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from .tnia_plotting_3d import show_xyz_max_slabs, show_xyz, create_multichannel_rgb, blend_colors, black_to
import importlib.resources as ir



class TNIAWidgetBase(anywidget.AnyWidget):
    # _esm = pathlib.Path(__file__).parent / "tnia_plotting_anywidgets.js"
    # _css = pathlib.Path(__file__).parent / "tnia_plotting_anywidgets.css"
    _esm = ir.files("eigenp_utils").joinpath("tnia_plotting_anywidgets.js")
    _css = ir.files("eigenp_utils").joinpath("tnia_plotting_anywidgets.css")


    # Data traits
    image_data = traitlets.Unicode(sync=True)

    # Sliders
    x_s = traitlets.Int(0).tag(sync=True)
    y_s = traitlets.Int(0).tag(sync=True)
    z_s = traitlets.Int(0).tag(sync=True)

    x_t = traitlets.Int(1).tag(sync=True)
    y_t = traitlets.Int(1).tag(sync=True)
    z_t = traitlets.Int(1).tag(sync=True)

    sxy = traitlets.Float(1.0).tag(sync=True)
    sz = traitlets.Float(1.0).tag(sync=True)

    # Bounds for sliders (computed)
    x_min_pos = traitlets.Int(0).tag(sync=True)
    x_max_pos = traitlets.Int(100).tag(sync=True)
    y_min_pos = traitlets.Int(0).tag(sync=True)
    y_max_pos = traitlets.Int(100).tag(sync=True)
    z_min_pos = traitlets.Int(0).tag(sync=True)
    z_max_pos = traitlets.Int(100).tag(sync=True)

    x_thick_max = traitlets.Int(100).tag(sync=True)
    y_thick_max = traitlets.Int(100).tag(sync=True)
    z_thick_max = traitlets.Int(100).tag(sync=True)

    min_thickness = traitlets.Int(1).tag(sync=True)

    # Save UI
    save_filename = traitlets.Unicode("filepath_save.svg").tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)

    def __init__(self, X, Y, Z, **kwargs):
        super().__init__(**kwargs)
        self.dims = (Z, Y, X) # (Z, Y, X) convention from numpy shape

        # Set max thickness bounds
        self.x_thick_max = max(1, X - 1)
        self.y_thick_max = max(1, Y - 1)
        self.z_thick_max = max(1, Z - 1)

    def _init_observers(self):
        # Observe thickness changes to update position bounds
        self.observe(self._update_bounds, names=['x_t', 'y_t', 'z_t'])

        # Observe all parameters to update plot
        self.observe(self._render_wrapper, names=['x_s', 'y_s', 'z_s', 'x_t', 'y_t', 'z_t'])

        # Observe save trigger
        self.observe(self._save_svg, names='save_trigger')

        # Initial bounds update
        self._update_bounds(None)

        # Initial render
        self._render_wrapper(None)

    def _update_bounds(self, change):
        # x
        lo = self.x_t
        hi = max(self.x_t, self.dims[2] - 1 - self.x_t)
        self.x_min_pos = lo
        self.x_max_pos = hi
        if self.x_s < lo: self.x_s = lo
        if self.x_s > hi: self.x_s = hi

        # y
        lo = self.y_t
        hi = max(self.y_t, self.dims[1] - 1 - self.y_t)
        self.y_min_pos = lo
        self.y_max_pos = hi
        if self.y_s < lo: self.y_s = lo
        if self.y_s > hi: self.y_s = hi

        # z
        lo = self.z_t
        hi = max(self.z_t, self.dims[0] - 1 - self.z_t)
        self.z_min_pos = lo
        self.z_max_pos = hi
        if self.z_s < lo: self.z_s = lo
        if self.z_s > hi: self.z_s = hi

    def _render_wrapper(self, change):
        fig = self._render()
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig) # Close to avoid memory leak

    def _render(self):
        raise NotImplementedError

    def _save_svg(self, change):
        fig = self._render()
        if fig:
            try:
                fig.savefig(self.save_filename, format='svg', dpi=300, bbox_inches='tight')
                print(f"Saved to {self.save_filename}")
            except Exception as e:
                print(f"Error saving file: {e}")
            finally:
                plt.close(fig)

class TNIASliceWidget(TNIAWidgetBase):
    def __init__(self, im, sxy=1, sz=1, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1, colors=None,
                 x_s=None, y_s=None, z_s=None, x_t=None, y_t=None, z_t=None):

        # Determine dimensions
        im_shape = (im[0].shape if isinstance(im, list) else im.shape)
        Z, Y, X = im_shape

        super().__init__(X, Y, Z)

        self.im = im
        self.sxy = sxy
        self.sz = sz
        self.figsize = figsize
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.gamma = gamma
        self.colors = colors

        # Set initial values if provided
        if x_t is not None: self.x_t = int(x_t)
        if y_t is not None: self.y_t = int(y_t)
        if z_t is not None: self.z_t = int(z_t)

        if x_s is not None: self.x_s = int(x_s)
        if y_s is not None: self.y_s = int(y_s)
        if z_s is not None: self.z_s = int(z_s)

        # Initialize observers and render
        self._init_observers()

    def _render(self):
        x_lims = [self.x_s - self.x_t, self.x_s + self.x_t]
        y_lims = [self.y_s - self.y_t, self.y_s + self.y_t]
        z_lims = [self.z_s - self.z_t, self.z_s + self.z_t]

        # Call original logic
        # Note: We need to handle show_crosshair manually or assume it's part of the original function?
        # show_xyz_max_slabs returns a Figure
        fig = show_xyz_max_slabs(
            self.im, x_lims, y_lims, z_lims,
            sxy=self.sxy, sz=self.sz, figsize=self.figsize, colormap=self.colormap,
            vmin=self.vmin, vmax=self.vmax, gamma=self.gamma, colors=self.colors
        )

        # Crosshairs logic (copied from original interactive wrapper)
        show_crosshair = True # Hardcoded or add parameter? Original wrapper defaulted to True
        if show_crosshair and fig:
            # XY
            fig.axes[0].axvline(x_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axvline(x_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[0].axhline(y_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            # ZY
            fig.axes[1].axvline(z_lims[0]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[0]*self.sxy + 0.5,     color='r', ls=':', alpha=0.3)
            fig.axes[1].axvline(z_lims[1]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[1].axhline(y_lims[1]*self.sxy + 0.5,     color='r', ls=':', alpha=0.3)
            # XZ
            fig.axes[2].axvline(x_lims[0]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[0]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)
            fig.axes[2].axvline(x_lims[1]*self.sxy + 0.5, color='r', ls=':', alpha=0.3)
            fig.axes[2].axhline(z_lims[1]*self.sz + 0.5*self.sz, color='r', ls=':', alpha=0.3)

        return fig


class TNIAScatterWidget(TNIAWidgetBase):
    def __init__(self, X_arr, Y_arr, Z_arr, channels=None, sxy=1, sz=1, render='points', bins=512,
                 point_size=4, alpha=0.6, colors=None, gamma=1, vmin=None, vmax=None, figsize=None,
                 x_s=None, y_s=None, z_s=None, x_t=None, y_t=None, z_t=None):

        self.X_arr = np.asarray(X_arr)
        self.Y_arr = np.asarray(Y_arr)
        self.Z_arr = np.asarray(Z_arr)

        # Compute bounds for init
        xmin, xmax = float(np.floor(self.X_arr.min())), float(np.ceil(self.X_arr.max()))
        ymin, ymax = float(np.floor(self.Y_arr.min())), float(np.ceil(self.Y_arr.max()))
        zmin, zmax = float(np.floor(self.Z_arr.min())), float(np.ceil(self.Z_arr.max()))

        X_dim = int(np.ceil(xmax - xmin + 1))
        Y_dim = int(np.ceil(ymax - ymin + 1))
        Z_dim = int(np.ceil(zmax - zmin + 1))

        super().__init__(X_dim, Y_dim, Z_dim)

        self.channels = channels
        self.sxy = sxy
        self.sz = sz
        self.render = render
        self.bins = bins
        self.point_size = point_size
        self.alpha = alpha
        self.colors = colors
        self.gamma = gamma
        self.vmin = vmin
        self.vmax = vmax
        self.figsize = figsize

        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax

        # Precompute Density helpers
        def _resolve_bins(B):
            if isinstance(B, (tuple, list)) and len(B) == 2:
                bx, by = int(B[0]), int(B[1])
            else:
                bx = by = int(B)
            bx = max(1, bx); by = max(1, by)
            return bx, by

        self.BX, self.BY = _resolve_bins(bins)

        # Analyze channels
        self.N = len(self.X_arr)

        def _is_int_like(a): return np.issubdtype(np.asarray(a).dtype, np.integer) or np.asarray(a).dtype == bool
        def _as_1d(a):
            a = np.asarray(a)
            return a.reshape(-1) if a.ndim == 1 else a

        self.mode = 'single'
        self.ch_ids = None
        self.cont_single = None
        self.cont_multi = None
        self.idx_lists = None

        if channels is None:
            self.mode = 'single'
        elif isinstance(channels, (list, tuple)):
            arrs = [ _as_1d(a) for a in channels ]
            lens = {len(a) for a in arrs}
            if lens == {self.N} and any(np.issubdtype(a.dtype, np.floating) for a in arrs):
                self.mode = 'cont_multi'
                self.cont_multi = np.stack([a.astype(float) for a in arrs], axis=1)
            else:
                self.mode = 'idx_lists'
                self.idx_lists = [np.asarray(ix, dtype=int) for ix in channels]
        else:
            a = _as_1d(channels)
            if _is_int_like(a):
                self.mode = 'ids'
                self.ch_ids = np.asarray(a, dtype=int)
            else:
                self.mode = 'cont_single'
                self.cont_single = np.asarray(a, dtype=float)

        if self.mode in ('single',):
            self.C = 1
        elif self.mode == 'ids':
            self.C = int(self.ch_ids.max()) + 1 if self.ch_ids.size else 1
        elif self.mode == 'idx_lists':
            self.C = len(self.idx_lists)
        elif self.mode == 'cont_single':
            self.C = 1
        else:
            self.C = self.cont_multi.shape[1]

        if self.colors is None:
            default_cols = ['magenta', 'cyan', 'yellow', 'green', 'red', 'lime', 'blue', 'orange']
            self.colors_use = default_cols[:max(1, self.C)]
        else:
            self.colors_use = self.colors

        self.colors_rgb = [matplotlib.colors.to_rgb(c) for c in self.colors_use]

        # Init values
        if x_t is not None: self.x_t = int(x_t)
        if y_t is not None: self.y_t = int(y_t)
        if z_t is not None: self.z_t = int(z_t)

        # Center default
        if x_s is None: self.x_s = int((self.xmax - self.xmin)/2) # relative to 0
        else: self.x_s = int(x_s)

        if y_s is None: self.y_s = int((self.ymax - self.ymin)/2)
        else: self.y_s = int(y_s)

        if z_s is None: self.z_s = int((self.zmax - self.zmin)/2)
        else: self.z_s = int(z_s)

        # Initialize observers and render
        self._init_observers()

    def _render(self):
        # Translate widget relative coordinates (0..Dim) to data coordinates (min..max)
        x_c = self.x_s + self.xmin
        y_c = self.y_s + self.ymin
        z_c = self.z_s + self.zmin

        x_lims = (x_c - self.x_t, x_c + self.x_t)
        y_lims = (y_c - self.y_t, y_c + self.y_t)
        z_lims = (z_c - self.z_t, z_c + self.z_t)

        # Density Mode Logic
        if self.render == 'density':
            EMPTY_XY = np.zeros((self.BY, self.BX), dtype=float)
            EMPTY_XZ = np.zeros((self.BY, self.BX), dtype=float)
            EMPTY_ZY = np.zeros((self.BY, self.BX), dtype=float)

            def _hist2d(x, y, xr, yr, w=None):
                H, _, _ = np.histogram2d(y, x, bins=[self.BY, self.BX], range=[yr, xr], weights=w)
                return H

            xy_list, xz_list, zy_list = [], [], []

            if self.mode in ('single', 'ids', 'idx_lists'):
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for idxs in idx_lists_local:
                    if idxs.size == 0:
                        xy_list.append(EMPTY_XY.copy()); xz_list.append(EMPTY_XZ.copy()); zy_list.append(EMPTY_ZY.copy()); continue
                    Xi, Yi, Zi = self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    Hxy = _hist2d(Xi[mZ], Yi[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1)) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(Xi[mY], Zi[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1)) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(Zi[mX], Yi[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1)) if mX.any() else EMPTY_ZY.copy()
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            elif self.mode == 'cont_single':
                vals = self.cont_single
                mZ = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
                mY = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
                mX = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

                Hxy = _hist2d(self.X_arr[mZ], self.Y_arr[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1), w=vals[mZ]) if mZ.any() else EMPTY_XY.copy()
                Hxz = _hist2d(self.X_arr[mY], self.Z_arr[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1), w=vals[mY]) if mY.any() else EMPTY_XZ.copy()
                Hzy = _hist2d(self.Z_arr[mX], self.Y_arr[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1), w=vals[mX]) if mX.any() else EMPTY_ZY.copy()

                xy_list = [Hxy]; xz_list = [Hxz]; zy_list = [Hzy]

            else: # cont_multi
                vals = self.cont_multi
                for c in range(self.C):
                    v = vals[:, c]
                    mZ = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
                    mY = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
                    mX = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

                    Hxy = _hist2d(self.X_arr[mZ], self.Y_arr[mZ], (self.xmin, self.xmax+1), (self.ymin, self.ymax+1), w=v[mZ]) if mZ.any() else EMPTY_XY.copy()
                    Hxz = _hist2d(self.X_arr[mY], self.Z_arr[mY], (self.xmin, self.xmax+1), (self.zmin, self.zmax+1), w=v[mY]) if mY.any() else EMPTY_XZ.copy()
                    Hzy = _hist2d(self.Z_arr[mX], self.Y_arr[mX], (self.zmin, self.zmax+1), (self.ymin, self.ymax+1), w=v[mX]) if mX.any() else EMPTY_ZY.copy()
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            if self.mode in ('single', 'ids', 'idx_lists', 'cont_multi'):
                colors_for_rgb = self.colors_use
            else:
                colors_for_rgb = self.colors_use # handled above logic in original? no, original handles 'cont_single' separately

            xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
                xy_list, xz_list, zy_list,
                vmin=self.vmin, vmax=self.vmax, gamma=self.gamma, colors=colors_for_rgb, blend='add', soft_clip=True
            )

            fig = show_xyz(
                xy_rgb, xz_rgb, zy_rgb,
                sxy=self.sxy, sz=self.sz, figsize=self.figsize, colormap=None,
                vmin=None, vmax=None, gamma=1, use_plt=True, colors=None
            )

            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("black")
            for ax in fig.axes:
                ax.set_facecolor("black")

            # Crosshairs
            show_crosshair = True
            if show_crosshair:
                axXY, axZY, axXZ = fig.axes[0], fig.axes[1], fig.axes[2]
                axXY.vlines([x_lims[0]*self.sxy + 0.5, (x_lims[1]+1)*self.sxy + 0.5], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axXY.hlines([y_lims[0]*self.sxy + 0.5, (y_lims[1]+1)*self.sxy + 0.5], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.vlines([z_lims[0]*self.sz + 0.5*self.sz, (z_lims[1]+1)*self.sz + 0.5*self.sz], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.hlines([y_lims[0]*self.sxy + 0.5,       (y_lims[1]+1)*self.sxy + 0.5], self.zmin*self.sz, (self.zmax+1)*self.sz, colors='r', linestyles=':', alpha=0.3)
                axXZ.vlines([x_lims[0]*self.sxy + 0.5, (x_lims[1]+1)*self.sxy + 0.5], self.zmin*self.sz, (self.zmax+1)*self.sz, colors='r', linestyles=':', alpha=0.3)
                axXZ.hlines([z_lims[0]*self.sz + 0.5*self.sz, (z_lims[1]+1)*self.sz + 0.5*self.sz], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)

            return fig

        else:
            # Points mode
            z_xy_ratio = (self.sz / self.sxy) if self.sxy != self.sz else 1
            width_ratios  = [int(self.xmax - self.xmin + 1), int((self.zmax - self.zmin + 1) * z_xy_ratio)]
            height_ratios = [int(self.ymax - self.ymin + 1), int((self.zmax - self.zmin + 1) * z_xy_ratio)]

            fig, axs = plt.subplots(
                2, 2, figsize=self.figsize, constrained_layout=False,
                gridspec_kw=dict(width_ratios=width_ratios, height_ratios=height_ratios),
                facecolor='black'
            )
            axXY, axZY = axs[0,0], axs[0,1]
            axXZ, axBar = axs[1,0], axs[1,1]
            for ax in (axXY, axZY, axXZ, axBar):
                ax.set_facecolor('black'); ax.axis('off')

            mZ_all = (self.Z_arr >= z_lims[0]) & (self.Z_arr <= z_lims[1])
            mY_all = (self.Y_arr >= y_lims[0]) & (self.Y_arr <= y_lims[1])
            mX_all = (self.X_arr >= x_lims[0]) & (self.X_arr <= x_lims[1])

            if self.mode == 'cont_single':
                vals = self.cont_single
                cmap = black_to(self.colors_use[0])
                norm = matplotlib.colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))

                if mZ_all.any():
                    axXY.scatter(self.X_arr[mZ_all]*self.sxy, self.Y_arr[mZ_all]*self.sxy, c=vals[mZ_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(self.X_arr[mY_all]*self.sxy, self.Z_arr[mY_all]*self.sz,  c=vals[mY_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(self.Z_arr[mX_all]*self.sz,  self.Y_arr[mX_all]*self.sxy, c=vals[mX_all], cmap=cmap, norm=norm,
                                 s=self.point_size, alpha=self.alpha, linewidths=0)

            elif self.mode == 'cont_multi':
                cols = blend_colors(self.cont_multi, self.colors_use, vmin=self.vmin, vmax=self.vmax, gamma=self.gamma, soft_clip=True)
                if mZ_all.any():
                    axXY.scatter(self.X_arr[mZ_all]*self.sxy, self.Y_arr[mZ_all]*self.sxy, c=cols[mZ_all], s=self.point_size, alpha=self.alpha, linewidths=0)
                if mY_all.any():
                    axXZ.scatter(self.X_arr[mY_all]*self.sxy, self.Z_arr[mY_all]*self.sz,  c=cols[mY_all], s=self.point_size, alpha=self.alpha, linewidths=0)
                if mX_all.any():
                    axZY.scatter(self.Z_arr[mX_all]*self.sz,  self.Y_arr[mX_all]*self.sxy, c=cols[mX_all], s=self.point_size, alpha=self.alpha, linewidths=0)

            else:
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for c, idxs in enumerate(idx_lists_local):
                    if idxs.size == 0: continue
                    Xi, Yi, Zi = self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs]
                    col = [self.colors_rgb[c % len(self.colors_rgb)]]
                    mZ = (Zi >= z_lims[0]) & (Zi <= z_lims[1])
                    mY = (Yi >= y_lims[0]) & (Yi <= y_lims[1])
                    mX = (Xi >= x_lims[0]) & (Xi <= x_lims[1])
                    if mZ.any(): axXY.scatter(Xi[mZ]*self.sxy, Yi[mZ]*self.sxy, s=self.point_size, c=col, alpha=self.alpha, linewidths=0)
                    if mY.any(): axXZ.scatter(Xi[mY]*self.sxy, Zi[mY]*self.sz,  s=self.point_size, c=col, alpha=self.alpha, linewidths=0)
                    if mX.any(): axZY.scatter(Zi[mX]*self.sz,  Yi[mX]*self.sxy, s=self.point_size, c=col, alpha=self.alpha, linewidths=0)

            # Axis limits
            axXY.set_xlim([self.xmin*self.sxy, (self.xmax+1)*self.sxy]); axXY.set_ylim([(self.ymax+1)*self.sxy, self.ymin*self.sxy])
            axXZ.set_xlim([self.xmin*self.sxy, (self.xmax+1)*self.sxy]); axXZ.set_ylim([(self.zmax+1)*self.sz,  self.zmin*self.sz ])
            axZY.set_xlim([self.zmin*self.sz,  (self.zmax+1)*self.sz ]); axZY.set_ylim([(self.ymax+1)*self.sxy, self.ymin*self.sxy])

            show_crosshair = True
            if show_crosshair:
                axXY.vlines([x_lims[0]*self.sxy, (x_lims[1]+1)*self.sxy], self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axXY.hlines([y_lims[0]*self.sxy, (y_lims[1]+1)*self.sxy], self.xmin*self.sxy, (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.vlines([z_lims[0]*self.sz,  (z_lims[1]+1)*self.sz],  self.ymin*self.sxy, (self.ymax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)
                axZY.hlines([y_lims[0]*self.sxy, (y_lims[1]+1)*self.sxy], self.zmin*self.sz,   (self.zmax+1)*self.sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.vlines([x_lims[0]*self.sxy, (x_lims[1]+1)*self.sxy], self.zmin*self.sz,   (self.zmax+1)*self.sz,  colors='r', linestyles=':', alpha=0.3)
                axXZ.hlines([z_lims[0]*self.sz,  (z_lims[1]+1)*self.sz],  self.xmin*self.sxy,  (self.xmax+1)*self.sxy, colors='r', linestyles=':', alpha=0.3)

            # Scale bar (kept opaque)
            fig.patch.set_alpha(1.0)
            width_um = (self.xmax - self.xmin + 1) * self.sxy
            target = width_um * 0.2
            def nice_length(x):
                exp = np.floor(np.log10(x))
                for m in [5,2,1]:
                    val = m * 10**exp
                    if val <= x: return val
                return x
            bar_um = nice_length(target)
            bar_pix = bar_um / self.sxy
            bar_frac = bar_pix / (self.xmax - self.xmin + 1)
            fig_h_in = self.figsize[1] if self.figsize else 10
            fontsize_pt = max(8, min(24, fig_h_in * 72 * 0.03))
            x0 = 0.5 - bar_frac/2; x1 = 0.5 + bar_frac/2; y = 0.5
            axBar.hlines(y, x0, x1, transform=axBar.transAxes, linewidth=2, color='gray')
            axBar.text(0.5, y - 0.1, f"{int(bar_um)} µm", transform=axBar.transAxes,
                       ha='center', va='top', color='gray', fontsize=fontsize_pt)

            fig.tight_layout(pad=0.0)
            return fig


def show_xyz_max_slice_interactive(
    im,
    sxy=1, sz=1,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True,
    colors=None,
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    im_shape = (im[0].shape if isinstance(im, list) else im.shape)
    Z, Y, X = im_shape
    z_xy_ratio = (sz / sxy) if sxy != sz else 1

    if figsize is None:
        width_px  = X + Z * z_xy_ratio
        height_px = Y + Z * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, n // 64)
    if x_t is None: x_t = _default_t(X)
    if y_t is None: y_t = _default_t(Y)
    if z_t is None: z_t = _default_t(Z)

    # Defaults for s are handled in init (midpoint)
    if x_s is None: x_s = X // 2
    if y_s is None: y_s = Y // 2
    if z_s is None: z_s = Z // 2

    return TNIASliceWidget(
        im, sxy=sxy, sz=sz, figsize=figsize, colormap=colormap,
        vmin=vmin, vmax=vmax, gamma=gamma, colors=colors,
        x_s=x_s, y_s=y_s, z_s=z_s, x_t=x_t, y_t=y_t, z_t=z_t
    )

def show_xyz_max_scatter_interactive(
    X, Y, Z,
    channels=None,
    sxy=1, sz=1,
    render='density',
    bins=512,
    point_size=4, alpha=0.6,
    colors=None,
    gamma=1, vmin=None, vmax=None,
    figsize=None, figsize_scale=1.0,
    show_crosshair=True,
    x_s=None, y_s=None, z_s=None,
    x_t=None, y_t=None, z_t=None,
):
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)

    xmin, xmax = float(np.floor(X.min())), float(np.ceil(X.max()))
    ymin, ymax = float(np.floor(Y.min())), float(np.ceil(Y.max()))
    zmin, zmax = float(np.floor(Z.min())), float(np.ceil(Z.max()))

    XN = xmax - xmin + 1
    YN = ymax - ymin + 1
    ZN = zmax - zmin + 1
    z_xy_ratio = (sz / sxy) if sxy != sz else 1

    if figsize is None:
        width_px  = XN + ZN * z_xy_ratio
        height_px = YN + ZN * z_xy_ratio
        divisor = max(width_px / 8, height_px / 8)
        w, h = float(width_px / divisor), float(height_px / divisor)
        figsize = (w * figsize_scale, h * figsize_scale)

    def _default_t(n): return max(1, int(n // 64))
    Xdim = int(np.ceil(XN)); Ydim = int(np.ceil(YN)); Zdim = int(np.ceil(ZN))

    if x_t is None: x_t = _default_t(Xdim)
    if y_t is None: y_t = _default_t(Ydim)
    if z_t is None: z_t = _default_t(Zdim)

    # Note: s inputs to scatter are in data coords (min-max range)
    # The Widget expects 0-Dim range?
    # Wait, my logic in TNIAScatterWidget._render adds xmin to x_s.
    # So x_s inside the widget is 0-based offset.
    # But x_s PASSED here is likely data coord?
    # Original code:
    # x_center_default = int(np.round((xmin + xmax) * 0.5))
    # x_s0 = _clip(int(x_s if x_s is not None else x_center_default), x_lo0, x_hi0)
    # So original input x_s is data coordinate.

    # So if x_s is provided, I need to subtract xmin to get 0-based offset for the widget init.
    if x_s is not None: x_s = int(x_s - xmin)
    if y_s is not None: y_s = int(y_s - ymin)
    if z_s is not None: z_s = int(z_s - zmin)

    return TNIAScatterWidget(
        X, Y, Z,
        channels=channels, sxy=sxy, sz=sz, render=render, bins=bins,
        point_size=point_size, alpha=alpha, colors=colors,
        gamma=gamma, vmin=vmin, vmax=vmax, figsize=figsize,
        x_s=x_s, y_s=y_s, z_s=z_s, x_t=x_t, y_t=y_t, z_t=z_t
    )
