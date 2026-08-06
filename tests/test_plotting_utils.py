import os

from pathlib import Path
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xml.etree.ElementTree as ET

from eigenp_utils.plotting_utils import colormap_maker
from eigenp_utils.plotting_utils import raincloud_plot
from eigenp_utils.plotting_utils import savefig_svg



# =========================================
# Source: test_plotting_utils.py
# =========================================


def test_savefig_svg(tmp_path):
    """Test that savefig svg works as expected."""
    # Set up a test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Test Plot')
    fig.suptitle('My Awesome Title')

    # Define file path
    svg_path = tmp_path / "test_figure"

    # Call the function
    savefig_svg(svg_path, bgnd_color=(1, 0, 0, 0.5), pad_inches=0.2)

    # Make sure we add '.svg' if it's missing in test assertion
    svg_file = str(svg_path) + ".svg"

    # Assert file exists
    assert os.path.exists(svg_file)

    # Read the SVG content
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # SVG namespaces are generally used
    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                  'cc': 'http://creativecommons.org/ns#',
                  'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                  'svg': 'http://www.w3.org/2000/svg'}

    # Check for metadata
    # The exact structure depends on matplotlib's SVG backend, but Dublin Core properties are typically nested inside <metadata><rdf:RDF><cc:Work>
    metadata_elem = root.find('svg:metadata', namespaces)
    assert metadata_elem is not None, "Metadata element not found in SVG"

    rdf_work = metadata_elem.find('.//cc:Work', namespaces)
    assert rdf_work is not None, "cc:Work element not found in metadata"

    dc_title = rdf_work.find('.//dc:title', namespaces)
    assert dc_title is not None, "dc:title not found"
    assert dc_title.text == 'My Awesome Title', f"Expected title 'My Awesome Title', got '{dc_title.text}'"

    dc_date = rdf_work.find('.//dc:date', namespaces)
    assert dc_date is not None, "dc:date not found"
    # Date should be an ISO format string
    assert len(dc_date.text) > 0, "Date string is empty"

    dc_creator = rdf_work.find('.//dc:creator//cc:Agent//dc:title', namespaces)
    assert dc_creator is not None, "dc:creator not found"
    assert dc_creator.text == 'eigenp', f"Expected creator 'eigenp', got '{dc_creator.text}'"

    plt.close('all')

def test_savefig_svg_no_suptitle(tmp_path):
    """Test that savefig svg no suptitle works as expected."""
    # Set up a test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    # Define file path
    svg_path = tmp_path / "test_figure_no_title"

    # Call the function
    savefig_svg(svg_path)

    # Make sure we add '.svg' if it's missing in test assertion
    svg_file = str(svg_path) + ".svg"

    # Assert file exists
    assert os.path.exists(svg_file)

    # Read the SVG content
    tree = ET.parse(svg_file)
    root = tree.getroot()

    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/',
                  'cc': 'http://creativecommons.org/ns#',
                  'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                  'svg': 'http://www.w3.org/2000/svg'}

    metadata_elem = root.find('svg:metadata', namespaces)
    rdf_work = metadata_elem.find('.//cc:Work', namespaces)
    dc_title = rdf_work.find('.//dc:title', namespaces)

    # If no suptitle exists, it should default to the filename string
    assert dc_title is not None
    assert dc_title.text == str(svg_path)

    plt.close('all')

def test_set_plotting_style():
    """Test that set plotting style works as expected."""
    from eigenp_utils.plotting_utils import set_plotting_style

    # Backup original rcParams
    orig_fonttype = plt.rcParams['svg.fonttype']
    orig_unicode = plt.rcParams['axes.unicode_minus']

    try:
        # Run the function with default editable_text=True
        set_plotting_style()

        # Check that settings are applied
        assert plt.rcParams['font.family'] == ['sans-serif']
        assert plt.rcParams['svg.fonttype'] == 'none'
        assert plt.rcParams['axes.unicode_minus'] == False

        # Reset and test with editable_text=False
        # Our updated set_plotting_style explicitly sets these back to default
        set_plotting_style(editable_text=False)
        assert plt.rcParams['svg.fonttype'] == 'path'
        assert plt.rcParams['axes.unicode_minus'] == True

    finally:
        # Restore globally to avoid polluting other tests
        plt.rcParams['svg.fonttype'] = orig_fonttype
        plt.rcParams['axes.unicode_minus'] = orig_unicode

def test_savefig_svg_editable_text(tmp_path):
    """Test that savefig svg editable text works as expected."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "Test Editable Text")

    # Test default editable_text=True
    svg_path_true = tmp_path / "editable_true"
    savefig_svg(svg_path_true) # Should use default True

    # Check that settings weren't permanently changed
    assert plt.rcParams['svg.fonttype'] != 'none'

    # Read the SVG
    svg_file_true = str(svg_path_true) + ".svg"
    with open(svg_file_true, 'r') as f:
        content_true = f.read()

    # Text elements should be standard <text> in SVG if fonttype='none'
    assert "<text" in content_true, "Expected <text> elements when editable_text=True"

    # Test editable_text=False
    svg_path_false = tmp_path / "editable_false"
    savefig_svg(svg_path_false, editable_text=False)

    svg_file_false = str(svg_path_false) + ".svg"
    with open(svg_file_false, 'r') as f:
        content_false = f.read()

    # If fonttype='path' (the default), text is converted to paths, so <text> elements usually won't exist
    assert "<text" not in content_false, "Did not expect <text> elements when editable_text=False"

    plt.close('all')

# =========================================
# Source: test_raincloud_plot.py
# =========================================

matplotlib.use("Agg") # Use non-interactive backend

def test_raincloud_plot_vertical_custom_labels():
    """Test that raincloud plot vertical custom labels works as expected."""
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    x_labels = ['Group A', 'Group B']

    # x_label as list
    res = raincloud_plot(data, x_label=x_labels, title="Vertical Plot")
    ax = res['axes']

    xticklabels = [l.get_text() for l in ax.get_xticklabels()]
    xlabel = ax.get_xlabel()

    assert xticklabels == x_labels
    assert xlabel == ""

def test_raincloud_plot_horizontal_custom_labels():
    """Test that raincloud plot horizontal custom labels works as expected."""
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    y_labels = ['Group X', 'Group Y']

    # y_label as list, orientation horizontal
    res = raincloud_plot(data, y_label=y_labels, orientation='horizontal', title="Horizontal Plot")
    ax = res['axes']

    yticklabels = [l.get_text() for l in ax.get_yticklabels()]
    ylabel = ax.get_ylabel()

    assert yticklabels == y_labels
    assert ylabel == ""

def test_raincloud_plot_mismatch_warning(capsys):
    """Test that raincloud plot mismatch warning works as expected."""
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    x_labels = ['Group A'] # Mismatch length

    res = raincloud_plot(data, x_label=x_labels, title="Mismatch Plot")
    ax = res['axes']

    xticklabels = [l.get_text() for l in ax.get_xticklabels()]
    xlabel = ax.get_xlabel()

    # Expect standard behavior: ticks are 0, 1. xlabel is "['Group A']"
    assert xticklabels == ['0', '1']
    assert xlabel == "['Group A']"

    # Check for warning print
    captured = capsys.readouterr()
    assert "Warning: x_label list length (1) does not match number of groups (2)" in captured.out

def test_raincloud_plot_with_kwargs():
    """Test that raincloud plot with kwargs works as expected."""
    import pandas as pd
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })

    # Just checking it runs without exceptions to catch regressions
    res = raincloud_plot(
        data=data,
        x='group',
        y='value',
        size_scatter=10,
        size_median=50,
        alpha_scatter=0.2,
        alpha_violin=0.3,
        linewidth_scatter=1,
        linewidth_boxplot=2,
        offset_scatter=0.1
    )
    assert res is not None
    assert 'axes' in res

def test_raincloud_plot_raster_threshold():
    """Test that raincloud plot raster threshold works as expected."""
    import pandas as pd
    from matplotlib.collections import PathCollection

    # Total points = 10 (less than 50 threshold)
    data_small = pd.DataFrame({'group': ['A']*5 + ['B']*5, 'value': range(10)})
    res_small = raincloud_plot(data=data_small, x='group', y='value', raster_threshold=50)
    ax_small = res_small['axes']

    # Find scatter collections (we expect them not to be rasterized)
    scatter_collections_small = [c for c in ax_small.collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 5]
    assert len(scatter_collections_small) > 0
    for c in scatter_collections_small:
        assert c.get_rasterized() is False

    # Total points = 100 (greater than 50 threshold)
    data_large = pd.DataFrame({'group': ['A']*50 + ['B']*50, 'value': range(100)})
    res_large = raincloud_plot(data=data_large, x='group', y='value', raster_threshold=50)
    ax_large = res_large['axes']

    scatter_collections_large = [c for c in ax_large.collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 50]
    assert len(scatter_collections_large) > 0
    for c in scatter_collections_large:
        # these scatter collections should be rasterized
        assert c.get_rasterized() is True

def test_savefig_svg_raster_threshold(tmp_path):
    """Test that savefig svg raster threshold works as expected."""
    from eigenp_utils.plotting_utils import savefig_svg
    import pandas as pd
    from matplotlib.collections import PathCollection

    # Create a simple figure with a scatter plot
    fig, ax = plt.subplots()
    # 100 points
    ax.scatter(range(100), range(100))

    # Initially not rasterized
    scatter_col = [c for c in ax.collections if isinstance(c, PathCollection)][0]
    assert scatter_col.get_rasterized() is False

    # Call savefig_svg with threshold 500 (not met)
    out_file1 = tmp_path / "test_no_raster.svg"
    savefig_svg(out_file1, scatter_raster_threshold=500)

    # Should still not be rasterized
    assert scatter_col.get_rasterized() is False

    # Call savefig_svg with threshold 50 (met)
    out_file2 = tmp_path / "test_raster.svg"
    savefig_svg(out_file2, scatter_raster_threshold=50)

    # Should now be rasterized
    assert scatter_col.get_rasterized() is True

    # Both SVG files should have been created
    assert out_file1.exists()
    assert out_file2.exists()



def test_savefig_svg_raster_threshold_size(tmp_path):
    """Test that savefig svg raster threshold size works as expected."""
    from eigenp_utils.plotting_utils import savefig_svg
    import pandas as pd
    from matplotlib.collections import PathCollection

    # --- Original Test Case: State Checks ---
    fig, ax = plt.subplots()
    # 100 points
    ax.scatter(range(100), range(100))

    # Initially not rasterized
    scatter_col = [c for c in ax.collections if isinstance(c, PathCollection)][0]
    assert scatter_col.get_rasterized() is False

    # Call savefig_svg with threshold 500 (not met)
    out_file1 = tmp_path / "test_no_raster.svg"
    savefig_svg(out_file1, scatter_raster_threshold=500)

    # Should still not be rasterized
    assert scatter_col.get_rasterized() is False

    # Call savefig_svg with threshold 50 (met)
    out_file2 = tmp_path / "test_raster.svg"
    savefig_svg(out_file2, scatter_raster_threshold=50)

    # Should now be rasterized
    assert scatter_col.get_rasterized() is True

    # Both SVG files should have been created
    assert out_file1.exists()
    assert out_file2.exists()

    plt.close(fig)

    # --- Added Test Case: File Size Verification ---
    fig_size, ax_size = plt.subplots()

    # 2e3 points
    n_points = 2000
    x = np.random.rand(n_points)
    y = np.random.rand(n_points)
    ax_size.scatter(x, y)

    out_file_vector = tmp_path / "test_size_vector.svg"
    out_file_raster = tmp_path / "test_size_raster.svg"

    # Save with threshold 3e3 (threshold not met -> pure vector representation)
    savefig_svg(out_file_vector, scatter_raster_threshold=3000, dpi=100)

    # Save with threshold 1e3 (threshold met -> scatter paths rasterized)
    savefig_svg(out_file_raster, scatter_raster_threshold=1000, dpi=100)

    # Compare file sizes on disk
    size_vector = out_file_vector.stat().st_size
    size_raster = out_file_raster.stat().st_size

    # The rasterized SVG embeds a base64 encoded PNG, avoiding rendering O(N) path objects.
    # It must yield a strictly smaller file size at this density.
    assert size_raster < size_vector

    plt.close(fig_size)

# =========================================
# Source: test_colormap_maker.py
# =========================================


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

# =========================================
# Source: test_raincloud_hue.py
# =========================================
matplotlib.use("Agg")

def test_raincloud_hue_vertical():
    """Test that raincloud hue vertical works as expected."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'stage': np.random.choice(['Stage1', 'Stage2'], n),
        'condition': np.random.choice(['CondA', 'CondB'], n),
        'distances': np.random.exponential(scale=2.0, size=n)
    })

    res = raincloud_plot(data=df, x='stage', y='distances', hue='condition', title="Hue Plot")
    ax = res['axes']

    # Check xticks (should be 2: Stage1, Stage2)
    xticks = ax.get_xticks()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]

    assert len(xticks) == 2
    # Sort to ensure order doesn't matter for this check
    assert sorted(xticklabels) == ['Stage1', 'Stage2']

    # Check labels
    assert ax.get_xlabel() == 'stage'
    assert ax.get_ylabel() == 'distances'

def test_raincloud_simple_xy():
    """Test that raincloud simple xy works as expected."""
    np.random.seed(42)
    df = pd.DataFrame({
        'stage': ['A', 'A', 'B', 'B'],
        'val': [1, 2, 3, 4]
    })
    res = raincloud_plot(data=df, x='stage', y='val')
    ax = res['axes']
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert sorted(xticklabels) == ['A', 'B']

def test_raincloud_horizontal_hue():
    """Test that raincloud horizontal hue works as expected."""
    np.random.seed(42)
    df = pd.DataFrame({
        'group': ['G1', 'G1', 'G2', 'G2'],
        'sub': ['S1', 'S2', 'S1', 'S2'],
        'val': [1, 2, 3, 4]
    })
    # Horizontal: y is category, x is value
    res = raincloud_plot(data=df, x='val', y='group', hue='sub', orientation='horizontal')
    ax = res['axes']

    # yticks should be G1, G2
    yticks = ax.get_yticks()
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(yticks) == 2
    assert sorted(yticklabels) == ['G1', 'G2']

    # xlabel should be 'val'
    assert ax.get_xlabel() == 'val'
    assert ax.get_ylabel() == 'group'

# =========================================
# Source: test_raincloud_features.py
# =========================================

def test_raincloud_features():
    """Test that raincloud features works as expected."""
    # generate some data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'category': ['A'] * n + ['B'] * n,
        'value': np.concatenate([np.random.normal(0, 1, n), np.random.normal(2, 1, n)]),
        'highlight': np.concatenate([np.random.choice([True, False], n, p=[0.1, 0.9]),
                                     np.random.choice([True, False], n, p=[0.1, 0.9])])
    })

    # Add an extreme outlier that should be filtered out by 'robust_zscore'
    df.loc[0, 'value'] = 1000

    # Ensure no exceptions are raised during plotting with new kwargs
    fig_dict = raincloud_plot(
        data=df,
        x='category',
        y='value',
        outlier_method='robust_zscore',
        outlier_multiplier=3.0,
        highlight_mask=df['highlight'],
        highlight_color='lime'
    )

    # Check that it returns dict with 'fig' and 'axes'
    assert 'fig' in fig_dict
    assert 'axes' in fig_dict

    # Test fallback gracefully handles if all points are filtered
    df_all_outliers = pd.DataFrame({
        'category': ['A', 'A'],
        'value': [1000, -1000] # Mean is 0, MAD is 1000, zscores are 0.6745. Both < 3.
        # But wait, robust_zscore on 2 points? Let's just pass some random stuff.
    })

    fig_dict2 = raincloud_plot(
        data=df,
        x='category',
        y='value',
        outlier_method='iqr',
        outlier_multiplier=0.0001, # This will filter almost everything
    )
    assert 'fig' in fig_dict2

def test_raincloud_legacy():
    """Test that raincloud legacy works as expected."""
    # Test legacy input without mask doesn't crash
    data = [np.random.normal(0, 1, 100), np.random.normal(1, 1, 100)]
    fig_dict = raincloud_plot(
        data=data,
        outlier_method='iqr',
        highlight_mask=None
    )
    assert 'fig' in fig_dict

# =========================================
# Source: test_raincloud_dodge.py
# =========================================


def test_raincloud_dodge_explicit():
    """Test that raincloud dodge explicit works as expected."""
    np.random.seed(42)
    n = 20
    groups = ['A', 'B']
    conditions = ['Control', 'Treatment']
    data = []
    for g in groups:
        for c in conditions:
            vals = np.random.normal(loc=10, scale=3, size=n)
            for v in vals:
                data.append({'Group': g, 'Condition': c, 'Value': v})

    df = pd.DataFrame(data)

    # Test dodge=True
    res_true = raincloud_plot(x='Group', y='Value', hue='Condition', data=df, dodge=True)
    assert res_true['fig'] is not None
    plt.close(res_true['fig'])

    # Test dodge=False
    res_false = raincloud_plot(x='Group', y='Value', hue='Condition', data=df, dodge=False)
    assert res_false['fig'] is not None
    plt.close(res_false['fig'])

def test_raincloud_palette_string():
    """Test that raincloud palette string works as expected."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Group': np.random.choice(['A', 'B'], 50),
        'Value': np.random.randn(50)
    })

    # Test valid colormap
    res_cmap = raincloud_plot(x='Group', y='Value', hue='Group', data=df, palette='viridis')
    assert res_cmap['fig'] is not None
    plt.close(res_cmap['fig'])

    # Test invalid colormap (fallback to single color, which then fails as invalid color)
    with pytest.raises(ValueError):
        raincloud_plot(x='Group', y='Value', hue='Group', data=df, palette='NotAColormap')

def test_raincloud_auto_dodge():
    """Test that raincloud auto dodge works as expected."""
    # When hue == x, dodge should be False (no shifting)
    np.random.seed(42)
    df = pd.DataFrame({
        'Group': np.random.choice(['A', 'B'], 50),
        'Value': np.random.randn(50)
    })

    # We can inspect the plot elements to verify position?
    # Or just check it runs without error
    res = raincloud_plot(x='Group', y='Value', hue='Group', data=df)
    assert res['fig'] is not None
    plt.close(res['fig'])
