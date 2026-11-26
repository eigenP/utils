"""Reference renders of the ortho viewers with µm-calibrated pixel sizes (D01 negative control).

``render_um_reference()`` draws the same small volumes through ``show_zyx_max``
(single channel and two channels) and ``TNIASliceWidget`` with explicit display
settings, so that only the pixel-size handling can change the output. The
fixture ``tnia_um_reference.npz`` next to this file was generated with this
function on the base commit af8d252 (before the shared conventions module):

    PYTHONPATH=<af8d252>/src python tests/_tnia_um_reference.py

Run from the repository root; it overwrites the fixture.
"""

import hashlib
import io
import pathlib

import matplotlib
import numpy as np

FIXTURE = pathlib.Path(__file__).with_name("tnia_um_reference.npz")


def _volume():
    rng = np.random.default_rng(42)
    vol = rng.integers(0, 1000, (6, 20, 30)).astype(np.uint16)
    vol[3, 5:9, 10:16] = 3000
    return vol


def _figure_record(fig, prefix):
    """Panel arrays, extents, axes positions and the scale-bar label of a figure."""
    out = {}
    for name in ("axXY", "axXZ", "axZY"):
        ax = getattr(fig, name)
        im = ax.images[0]
        out[f"{prefix}_{name}_array"] = np.asarray(im.get_array(), dtype=float)
        out[f"{prefix}_{name}_extent"] = np.asarray(im.get_extent(), dtype=float)
        out[f"{prefix}_{name}_position"] = np.asarray(ax.get_position().bounds, dtype=float)
        if im.get_array().ndim == 2:
            out[f"{prefix}_{name}_clim"] = np.asarray(im.get_clim(), dtype=float)
    out[f"{prefix}_label"] = np.array([t.get_text() for t in fig.axXY.texts])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=40)
    out[f"{prefix}_png_sha256"] = np.array(hashlib.sha256(buf.getvalue()).hexdigest())
    return out


def render_um_reference():
    """Dict of reference arrays (see module docstring)."""
    import matplotlib.pyplot as plt
    from eigenp_utils.tnia_plotting_anywidgets import TNIASliceWidget, show_zyx_max

    vol = _volume()
    ps = {"Z": 2.0, "Y": 0.5, "X": 0.5, "unit": "µm"}
    rec = {"matplotlib_version": np.array(matplotlib.__version__)}

    fig = show_zyx_max(vol, pixel_sizes=ps, figsize=(4, 4), colormap="gray", vmin=0, vmax=3000)
    rec.update(_figure_record(fig, "single"))
    plt.close(fig)

    other = (vol[:, ::-1, :] // 2).copy()
    fig = show_zyx_max([vol, other], pixel_sizes=(2.0, 0.5, 0.5), figsize=(4, 4),
                       colormap=["magenta", "lime"], vmin=[0, 0], vmax=[3000, 1500])
    rec.update(_figure_record(fig, "multi"))
    plt.close(fig)

    # The channel-label font is pinned to the base commit's automatic size for
    # a 4-inch figure (4 * 72 * 0.035 pt): R1-23 later changed the automatic
    # size, which is not what this pixel-size control is about.
    w = TNIASliceWidget(vol, pixel_sizes=ps, figsize=(4, 4), colormap="white", vmin=0, vmax=2500,
                        channel_label_fontsize_pt=4 * 72 * 0.035)
    fig = w._render()
    rec.update(_figure_record(fig, "widget"))
    plt.close(fig)
    return rec


if __name__ == "__main__":
    np.savez_compressed(FIXTURE, **render_um_reference())
    print(f"wrote {FIXTURE}")
