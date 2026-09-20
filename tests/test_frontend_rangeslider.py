import functools
import http.server
from pathlib import Path
import tempfile
import threading
import pytest

from eigenp_utils.tnia_plotting_anywidgets import TNIAWidgetBase

@pytest.fixture(scope="module")
def static_rangeslider_server():
    """Spins up a lightweight HTTP server to host the range slider JS frontend."""
    tmpdir = tempfile.TemporaryDirectory()
    base_path = Path(tmpdir.name)

    js_path = base_path / "tnia_plotting_anywidgets.js"
    js_path.write_text(TNIAWidgetBase._esm)

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: sans-serif; padding: 20px; }
      </style>
    </head>
    <body>
      <div id="widget-container"></div>
      <script type="module">
        import widget from './tnia_plotting_anywidgets.js';

        const listeners = {};

        window.mockModel = {
          state: {
            image_data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            x_start: 20, x_end: 60, x_max_pos: 100, sx: 1.0,
            y_start: 30, y_end: 70, y_max_pos: 100, sy: 1.0,
            z_start: 10, z_end: 50, z_max_pos: 100, sz: 1.0,
            channel_names: ["DAPI"],
            channel_dtypes: ["uint8"],
            channel_colors: ["#00ffff"],
            vmin_list: [0],
            vmax_list: [255],
            gamma_list: [1.0],
            opacity_list: [1.0],
            histograms_data: [],
            show_crosshair: true,
            sync_on_hover: false,
            warning_msg: '',
            save_filename: 'plot.svg'
          },
          get(key) { return this.state[key]; },
          set(key, val) {
            this.state[key] = val;
            if (listeners[`change:${key}`]) {
              listeners[`change:${key}`].forEach(cb => cb());
            }
          },
          save_changes() { window.saveTriggered = true; },
          on(evt, cb) {
            if (!listeners[evt]) listeners[evt] = [];
            listeners[evt].push(cb);
          }
        };

        widget.render({ model: window.mockModel, el: document.getElementById('widget-container') });
      </script>
    </body>
    </html>
    """
    (base_path / "index.html").write_text(html_content)

    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(base_path))
    httpd = http.server.HTTPServer(('localhost', 0), Handler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}"

    httpd.shutdown()
    tmpdir.cleanup()

def test_frontend_rangeslider_ui(page, static_rangeslider_server):
    """Uses Playwright to visually inspect and test range slider handles and range translation."""
    page.goto(static_rangeslider_server)

    # Wait for the range sliders to render in the DOM
    x_range_label = page.get_by_text("X Range")
    x_range_label.wait_for(state="visible")

    # Locate middle (translate) thumb
    translate_thumb = page.locator("div[title='Translate Range']").first
    assert translate_thumb.is_visible()

    # Get initial values from mockModel
    initial_x_start = page.evaluate("window.mockModel.get('x_start')")
    initial_x_end = page.evaluate("window.mockModel.get('x_end')")
    assert initial_x_start == 20
    assert initial_x_end == 60

    # Drag the middle translate thumb horizontally
    box = translate_thumb.bounding_box()
    assert box is not None
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2

    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(start_x + 50, start_y)
    page.mouse.up()

    # Verify that range translated (both x_start and x_end moved by equal delta)
    new_x_start = page.evaluate("window.mockModel.get('x_start')")
    new_x_end = page.evaluate("window.mockModel.get('x_end')")

    assert new_x_start > initial_x_start
    assert new_x_end > initial_x_end
    # Range span should remain constant (60 - 20 = 40)
    assert (new_x_end - new_x_start) == (initial_x_end - initial_x_start)
