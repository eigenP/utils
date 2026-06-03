1. Update JS side in `src/eigenp_utils/tnia_plotting_anywidgets.js`:
    - Add `copyParamsBtn` button next to the SVG save button.
    - Set up event listener on `copyParamsBtn` to increment `copy_params_trigger` traitlet.
    - Set up listener for `copy_params_string` traitlet to write text to clipboard.
2. Update base widget python side `TNIAWidgetBase` in `src/eigenp_utils/tnia_plotting_anywidgets.py`:
    - Add traitlets: `copy_params_trigger` (Int) and `copy_params_string` (Unicode).
    - In `_init_observers`, add an observer for `copy_params_trigger`.
    - Implement `_copy_params(self, change)` method that grabs parameter values, formats them, and sets `copy_params_string`.
3. Update Slice Widget (`TNIASliceWidget._render`):
    - When drawing crosshairs, if rotation is not zero, calculate the rotated start and end points using the correct transformation (skimage.transform.rotate rotates the image counter-clockwise around its center), and use `plot([x0, x1], [y0, y1])` instead of `axvline/axhline`.
4. Update Scatter Widget (`TNIAScatterWidget._render`):
    - When drawing crosshairs in points mode, if rotation is not zero, rotate the points the same way it does for scatter plots (`_rotate_points_2d`) and draw them.
5. In both Slice and Scatter, update the `if` conditions to NOT bypass crosshairs when `rotate_view` is present.
6. Verify changes visually with playwright and via test scripts.
