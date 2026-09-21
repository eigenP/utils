import functools
import http.server
from pathlib import Path
import tempfile
import threading
import pytest

from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
from conftest import create_sham_volume

@pytest.fixture(scope="module")
def static_rangeslider_server():
    """Spins up a lightweight HTTP server to host the range slider JS frontend with sham volume."""
    tmpdir = tempfile.TemporaryDirectory()
    base_path = Path(tmpdir.name)

    vol = create_sham_volume() # (2, 64, 128, 128)
    im_list = [vol[0], vol[1]]
    w = show_zyx_max_slice_interactive(im_list, colormap=['magenta', 'green'], channel_labels=['Cube', 'Circle'])

    js_path = base_path / "tnia_plotting_anywidgets.js"
    js_path.write_text(w._esm)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{ font-family: sans-serif; padding: 20px; }}
      </style>
    </head>
    <body>
      <div id="widget-container"></div>
      <script type="module">
        import widget from './tnia_plotting_anywidgets.js';

        const listeners = {{}};

        window.mockModel = {{
          state: {{
            image_data: '{w.image_data}',
            x_s: {w.x_s}, y_s: {w.y_s}, z_s: {w.z_s},
            x_t: {w.x_t}, y_t: {w.y_t}, z_t: {w.z_t},
            x_start: {w.x_start}, x_end: {w.x_end}, x_max_pos: {w.x_max_pos}, sx: {w.sx},
            y_start: {w.y_start}, y_end: {w.y_end}, y_max_pos: {w.y_max_pos}, sy: {w.sy},
            z_start: {w.z_start}, z_end: {w.z_end}, z_max_pos: {w.z_max_pos}, sz: {w.sz},
            channel_names: {w.channel_names},
            channel_dtypes: {w.channel_dtypes},
            channel_colors: {w.channel_colors},
            vmin_list: {w.vmin_list},
            vmax_list: {w.vmax_list},
            gamma_list: {w.gamma_list},
            opacity_list: {w.opacity_list},
            histograms_data: {w.histograms_data},
            show_crosshair: {str(w.show_crosshair).lower()},
            sync_on_hover: {str(w.sync_on_hover).lower()},
            warning_msg: '{w.warning_msg}',
            save_filename: '{w.save_filename}'
          }},
          get(key) {{ return this.state[key]; }},
          set(key, val) {{
            this.state[key] = val;
            if (listeners[`change:${{key}}`]) {{
              listeners[`change:${{key}}`].forEach(cb => cb());
            }}
          }},
          save_changes() {{ window.saveTriggered = true; }},
          on(evt, cb) {{
            if (!listeners[evt]) listeners[evt] = [];
            listeners[evt].push(cb);
          }}
        }};

        widget.render({{ model: window.mockModel, el: document.getElementById('widget-container') }});
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
    """Uses Playwright to visually inspect and test range slider handles and range translation on sham volume."""
    page.goto(static_rangeslider_server)

    x_range_label = page.get_by_text("X Range")
    x_range_label.wait_for(state="visible")

    translate_thumb = page.locator("div[title='Translate Range']").first
    assert translate_thumb.is_visible()

    initial_x_start = page.evaluate("window.mockModel.get('x_start')")
    initial_x_end = page.evaluate("window.mockModel.get('x_end')")

    box = translate_thumb.bounding_box()
    assert box is not None
    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2

    page.mouse.move(start_x, start_y)
    page.mouse.down()
    page.mouse.move(start_x + 30, start_y)
    page.mouse.up()

    new_x_start = page.evaluate("window.mockModel.get('x_start')")
    new_x_end = page.evaluate("window.mockModel.get('x_end')")

    assert new_x_start > initial_x_start
    assert new_x_end > initial_x_end
    assert (new_x_end - new_x_start) == (initial_x_end - initial_x_start)
