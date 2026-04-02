def test_tnia_figsize_scale():
    import numpy as np
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    im = np.random.rand(100, 100, 100)
    widget1 = show_zyx_max_slice_interactive(im, figsize_scale=1)
    widget2 = show_zyx_max_slice_interactive(im, figsize_scale=2)
    widget3 = show_zyx_max_slice_interactive(im, figsize_scale=10)
    print(widget1.figsize)
    print(widget2.figsize)
    print(widget3.figsize)
