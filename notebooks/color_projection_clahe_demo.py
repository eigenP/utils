import marimo as mo

__generated_with__ = "0.6.15"

app = mo.App()


@app.cell
def __():
    import marimo as mo

    mo.md(
        """
        # Color-coded projection and CLAHE demo

        This notebook demonstrates how to use the
        `color_coded_projection` and `_my_clahe_` utilities provided in this
        repository. We load the sample `cells3d` dataset from scikit-image and
        showcase both functions:

        * **color_coded_projection** for creating a time/volume color projection
        * **_my_clahe_** for applying Contrast Limited Adaptive Histogram Equalization (CLAHE)

        Use the controls below to explore different color mappings for the
        projection and adjust the CLAHE clip limit to see its effect on the
        enhanced slice.
        """
    )


@app.cell
def __():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data

    from clahe_equalize_adapthist import _my_clahe_
    from color_coded_projection import color_coded_projection

    return plt, np, data, _my_clahe_, color_coded_projection


@app.cell
def __():
    import marimo as mo

    colormap_dropdown = mo.ui.dropdown(
        label="Projection colormap",
        options=[
            ("Plasma", "plasma"),
            ("Viridis", "viridis"),
            ("Inferno", "inferno"),
            ("Magma", "magma"),
            ("Cividis", "cividis"),
        ],
        value="plasma",
    )

    colormap_dropdown

    return colormap_dropdown


@app.cell
def __():
    import marimo as mo

    clahe_clip_slider = mo.ui.slider(
        label="CLAHE clip limit",
        start=0.01,
        stop=0.1,
        step=0.005,
        value=0.03,
    )

    clahe_clip_slider

    return clahe_clip_slider


@app.cell
def __(data):
    cells = data.cells3d()
    # Select the membrane channel (index 1)
    membrane_stack = cells[:, 1, :, :]
    return membrane_stack


@app.cell
def __(membrane_stack, np):
    # Normalize the stack to the range [0, 1]
    stack_min = membrane_stack.min()
    stack_max = membrane_stack.max()
    normalized_stack = (membrane_stack - stack_min) / (stack_max - stack_min)
    return normalized_stack


@app.cell
def __(color_coded_projection, colormap_dropdown, normalized_stack, np):
    projection = color_coded_projection(
        normalized_stack.astype(np.float32),
        color_map=colormap_dropdown.value,
    )
    return projection


@app.cell
def __(colormap_dropdown, projection, plt):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(projection)
    ax.set_title(
        f"Color-coded projection of membrane channel (cmap: {colormap_dropdown.value})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig


@app.cell
def __(membrane_stack):
    slice_index = 30
    original_slice = membrane_stack[slice_index]
    return original_slice, slice_index


@app.cell
def __(_my_clahe_, clahe_clip_slider, original_slice):
    clahe_slice = _my_clahe_(
        original_slice,
        clip_limit=float(clahe_clip_slider.value),
        nbins=256,
    )
    return clahe_slice


@app.cell
def __(clahe_clip_slider, clahe_slice, original_slice, plt, slice_index):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title(f"Original slice {slice_index}")
    axes[0].axis("off")

    axes[1].imshow(clahe_slice, cmap="gray")
    axes[1].set_title(
        f"CLAHE enhanced slice (clip_limit={float(clahe_clip_slider.value):.3f})"
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig


if __name__ == "__main__":
    app.run()
