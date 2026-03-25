def test_tnia_annotator_widget():
    from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget
    import numpy as np
    import os

    os.environ['TEST_DIR'] = '/tmp'

    im = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
    w = TNIAAnnotatorWidget(im)
    w.points = [[5, 5, 5], [6, 6, 6]]
    w.save_csv_filename = "$TEST_DIR/test_points.csv"
    w._save_csv(None)

    with open("/tmp/test_points.csv", "r") as f:
        print(f.read())
