
import marimo as mo

app = mo.App()

@app.cell
def __():
    import numpy as np
    from skimage.data import cells3d
    from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive

    im = cells3d()  # (Z, C, Y, X)
    membrane = im[:, 0, :, :]
    nuclei = im[:, 1, :, :]

    widget = show_xyz_max_slice_interactive(
        [membrane, nuclei],
        colors=['magma', 'viridis']
    )
    return widget,

if __name__ == "__main__":
    app.run()
