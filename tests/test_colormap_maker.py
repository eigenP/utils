import pytest
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from eigenp_utils.plotting_utils import colormap_maker

def test_colormap_maker_basic():
    """Test creating a colormap without positions and without registering it."""
    colors = ['k', 'cyan', (1.0, 1.0, 1.0, 0.5)]
    cmap = colormap_maker(colors)

    assert cmap is not None
    assert cmap.name == "custom_cmap"
    assert cmap.N == 256

    # Check that it's NOT registered since cmap_name is None
    with pytest.raises(ValueError):
        plt.get_cmap("custom_cmap")

def test_colormap_maker_with_positions():
    """Test creating a colormap with positions."""
    colors = ['red', 'green', 'blue']
    positions = [0.0, 0.2, 1.0]
    cmap = colormap_maker(colors, positions=positions)

    assert cmap is not None
    assert cmap.name == "custom_cmap"

    # Check that positions are correctly mapped.
    # Color at 0.0 should be red
    assert cmap(0.0) == mpl.colors.to_rgba('red')
    # Color at 0.2 should be green
    assert cmap(0.2) == mpl.colors.to_rgba('green')
    # Color at 1.0 should be blue
    assert cmap(1.0) == mpl.colors.to_rgba('blue')

def test_colormap_maker_registration():
    """Test registering a colormap with a custom name."""
    colors = ['#08041c', '#390b5e', '#a2217c', '#f04e4c', '#fce205']
    cmap_name = 'synthwave'
    cmap = colormap_maker(colors, cmap_name=cmap_name)

    assert cmap is not None
    assert cmap.name == cmap_name

    # Check that it is registered
    registered_cmap = plt.get_cmap(cmap_name)
    assert registered_cmap is not None
    assert registered_cmap.name == cmap_name

    # Clean up registration after test (if possible in this version of mpl)
    if hasattr(mpl.colormaps, 'unregister'):
        mpl.colormaps.unregister(cmap_name)

def test_colormap_maker_validation():
    """Test that colormap_maker validates inputs correctly."""
    colors = ['black', 'white']

    # Number of positions must match number of colors
    with pytest.raises(ValueError, match="The number of positions must match"):
        colormap_maker(colors, positions=[0.0])

    # Positions must start with 0.0 and end with 1.0
    with pytest.raises(ValueError, match="Positions must start with 0.0 and end with 1.0"):
        colormap_maker(colors, positions=[0.1, 1.0])

    with pytest.raises(ValueError, match="Positions must start with 0.0 and end with 1.0"):
        colormap_maker(colors, positions=[0.0, 0.9])

    # Positions must be strictly monotonically increasing
    colors4 = ['red', 'green', 'blue', 'yellow']
    with pytest.raises(ValueError, match="Positions must be strictly monotonically increasing"):
        colormap_maker(colors4, positions=[0.0, 0.6, 0.4, 1.0])

def test_cool_colormap():
    """Test creating a cool colormap like cyberpunk or retro wave."""
    # A dark neon colormap
    neon_colors = ['#08041c', '#390b5e', '#a2217c', '#f04e4c', '#fce205']
    positions = [0.0, 0.2, 0.5, 0.8, 1.0]
    cmap_name = 'cyberpunk_neon'

    cmap = colormap_maker(neon_colors, positions=positions, cmap_name=cmap_name)

    assert cmap is not None
    assert cmap.name == cmap_name

    # Generate some sample data
    data = np.random.rand(10, 10)

    # Make sure we can use the colormap name in matplotlib plotting
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap_name)

    # Asserting the mapping was used
    assert im.cmap.name == cmap_name
    plt.close(fig)
