
    import marimo as mo

    app = mo.App()

    @app.cell
    def __():
        import numpy as np
        from skimage.data import cells3d
        from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive

        try:
            im = cells3d()
        except:
            from eigenp_utils.io import download_file
            url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
            download_file(url_to_fetch, "./cells3d.tif")
            from skimage.io import imread
            im = imread("./cells3d.tif")  # (Z, C, Y, X)
            membrane = im[:, 0, :, :]
            nuclei = im[:, 1, :, :]
    
        widget = show_xyz_max_slice_interactive(
                [membrane, nuclei],
                colors=['magma', 'viridis']
            )
        return widget,

    if __name__ == "__main__":
        app.run()
