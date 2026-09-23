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

        // Exposed so tests can assert the widget does not accumulate
        // listeners when it rebuilds parts of its UI.
        window.listenerCount = (evt) => (listeners[evt] || []).length;

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


def test_frontend_rangeslider_integer_step_on_both_inputs(page, static_rangeslider_server):
    """Both numeric boxes of a range slider must expose an integer step when unscaled.

    The fixture volume is built without `pixel_sizes`, so every scale trait is
    1.0 and the boxes hold raw voxel indices. Strategy: read the `step`
    attribute the widget assigned to each number input. A start box stepping by
    1 while its end box steps by 0.01 (or carries no step at all) lets the
    spinner arrows produce fractional slab bounds on one side only, so every
    box is expected to report a step of exactly "1".
    """
    page.goto(static_rangeslider_server)
    page.get_by_text("X Range").wait_for(state="visible")

    steps = page.evaluate(
        "() => [...document.querySelectorAll('input[type=number]')].map(i => i.step)"
    )

    # Three sliders (X, Y, Z), each with a start and an end box.
    assert len(steps) == 6
    assert set(steps) == {"1"}


def test_frontend_channel_rebuild_does_not_leak_listeners(page, static_rangeslider_server):
    """Rebuilding the channel rows must not accumulate model listeners.

    `updateChannelsUI` discards and recreates every channel row whenever
    `channel_names` or `channel_colors` changes. The model exposes no way to
    unsubscribe, so any listener registered per rebuild survives forever and
    keeps firing against detached DOM nodes -- a leak that grows without bound
    as channels are renamed or recoloured. Strategy: drive three rebuilds and
    assert the per-trait listener counts are unchanged, then confirm the rows
    that survived are still wired to the model by pushing a new `gamma_list`
    through and reading it back out of the inputs.
    """
    page.goto(static_rangeslider_server)
    page.get_by_text("X Range").wait_for(state="visible")

    channel_traits = [
        "vmin_list", "vmax_list", "gamma_list", "opacity_list", "histograms_data",
    ]
    before = {t: page.evaluate(f"window.listenerCount('change:{t}')") for t in channel_traits}
    assert all(count > 0 for count in before.values()), before

    for names in (["Cube", "Circle"], ["Alpha", "Beta"], ["Gamma", "Delta"]):
        page.evaluate("names => window.mockModel.set('channel_names', names)", names)

    after = {t: page.evaluate(f"window.listenerCount('change:{t}')") for t in channel_traits}
    assert after == before

    # The rows standing after the last rebuild must still follow the model.
    page.evaluate("window.mockModel.set('gamma_list', [0.25, 0.75])")
    gamma_values = page.evaluate(
        "() => [...document.querySelectorAll('span')]"
        ".filter(s => s.textContent === 'gamma')"
        ".map(s => s.nextElementSibling.value)"
    )
    assert gamma_values == ["0.25", "0.75"]
