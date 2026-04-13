import pytest
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slabs, show_zyx_max_slice_interactive

def test_show_zyx_max_slabs_scale_bar():
    im = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)

    # Using tuple
    fig2 = show_zyx_max_slabs(im, pixel_sizes=(1.5, 0.5, 0.5))

    # Using dictionary
    fig3 = show_zyx_max_slabs(im, pixel_sizes={'Z': 1.5, 'Y': 0.5, 'X': 0.5})

    def get_text_from_fig(fig):
        axBar = fig.axes[-1]
        for t in axBar.texts:
            if "µm" in t.get_text() or "pixel_sizes" in t.get_text() or "sxy" in t.get_text():
                return t.get_text()
        return None

    # Assert
    assert "µm" in get_text_from_fig(fig2)
    assert "µm" in get_text_from_fig(fig3)

def test_interactive_factory_passes_pixel_sizes():
    im = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)

    # Should not throw errors and initialize perfectly
    widget = show_zyx_max_slice_interactive(im, pixel_sizes=(1.5, 0.5, 0.5))
    assert widget._pixel_sizes_given is True
    assert widget.sx == 0.5
    assert widget.sy == 0.5
    assert widget.sz == 1.5

if __name__ == '__main__':
    test_show_zyx_max_slabs_scale_bar()
    test_interactive_factory_passes_pixel_sizes()
    print("Tests passed")
