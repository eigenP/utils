
import numpy as np
import ipywidgets
from eigenp_utils.tnia_plotting_3d import show_xyz_max_slice_interactive, show_xyz_max_scatter_interactive

def test_slice_interactive_ui():
    # Create a dummy 3D image
    im = np.random.rand(10, 10, 10)

    # Call the function
    widget = show_xyz_max_slice_interactive(im)

    # Verify it returns a VBox (wrapper)
    assert isinstance(widget, ipywidgets.VBox), "Should return a VBox"

    # Check children: interactive widget + HBox (text + button)
    children = widget.children
    assert len(children) == 2, "Should have 2 children: interactive widget and save controls"

    interactive_widget = children[0]
    save_controls = children[1]

    # Verify interactive widget
    assert isinstance(interactive_widget, ipywidgets.interactive), "First child should be interactive widget"

    # Verify save controls
    assert isinstance(save_controls, ipywidgets.HBox), "Second child should be HBox"
    assert len(save_controls.children) == 2, "Save controls should have text and button"

    text_box = save_controls.children[0]
    button = save_controls.children[1]

    assert isinstance(text_box, ipywidgets.Text), "First control should be Text"
    assert text_box.value == 'filepath_save.svg', "Default filename should be correct"

    assert isinstance(button, ipywidgets.Button), "Second control should be Button"
    assert button.description == 'Save as SVG', "Button label should be correct"

def test_scatter_interactive_ui():
    # Create dummy scatter data
    X = np.random.rand(100) * 10
    Y = np.random.rand(100) * 10
    Z = np.random.rand(100) * 10

    # Call the function
    widget = show_xyz_max_scatter_interactive(X, Y, Z)

    # Verify it returns a VBox (wrapper)
    assert isinstance(widget, ipywidgets.VBox), "Should return a VBox"

    # Check structure (same as above)
    children = widget.children
    assert len(children) == 2
    assert isinstance(children[0], ipywidgets.interactive)
    assert isinstance(children[1], ipywidgets.HBox)

    save_controls = children[1]
    text_box = save_controls.children[0]
    button = save_controls.children[1]

    assert isinstance(text_box, ipywidgets.Text)
    assert isinstance(button, ipywidgets.Button)

if __name__ == "__main__":
    test_slice_interactive_ui()
    test_scatter_interactive_ui()
    print("UI structure verification passed!")
