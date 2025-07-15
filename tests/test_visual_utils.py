import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import colormaps as mpl_colormaps
from matplotlib.figure import Figure
from color_coded_projection import color_coded_projection
from hist_imshow import hist_imshow


def test_color_coded_projection_basic():
    img = np.zeros((3, 2, 2), dtype=float)
    img[1] = 1.0
    result = color_coded_projection(img, color_map='plasma')
    assert result.shape == (2, 2, 3)
    expected_color = mpl_colormaps['plasma'](1 / (3 - 1))[:3]
    expected = np.stack([np.full((2, 2), expected_color[c]) for c in range(3)], axis=-1)
    assert np.allclose(result, expected)


def test_color_coded_projection_invalid_input():
    with pytest.raises(ValueError):
        color_coded_projection(np.zeros((2, 2)))


def test_hist_imshow_return_image_only():
    img = np.random.rand(3, 4, 5)
    slice_img = hist_imshow(img, return_image_only=True)
    assert slice_img.shape == (4, 5)
    assert np.allclose(slice_img, img[1])


def test_hist_imshow_returns_figure():
    fig = hist_imshow(np.random.rand(2, 2))
    assert isinstance(fig, Figure)


def test_hist_imshow_xlabel_contains_statistics():
    img = np.array([[0, 1], [2, 3]], dtype=float)
    fig = hist_imshow(img)
    label = fig.axes[1].get_xlabel()
    expected = f"min:{img.min():.3g} max:{img.max():.3g} mean:{img.mean():.3g}"
    assert expected in label
