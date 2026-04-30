
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend
import matplotlib.pyplot as plt
from eigenp_utils.plotting_utils import raincloud_plot

def test_raincloud_plot_vertical_custom_labels():
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
