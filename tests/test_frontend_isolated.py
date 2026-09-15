import pytest
import http.server
import functools
import threading
import tempfile
from pathlib import Path
from eigenp_utils.tnia_plotting_anywidgets import TNIAWidgetBase

@pytest.fixture(scope="module")
def static_server():
    """Spins up a lightweight, built-in HTTP server to host the JS module."""
    tmpdir = tempfile.TemporaryDirectory()
    base_path = Path(tmpdir.name)

    # 1. Export the exact JS from your Python class
    js_path = base_path / "tnia_plotting_anywidgets.js"
    js_path.write_text(TNIAWidgetBase._esm)

    # 2. Create the mock DOM and AnyWidget model harness
    html_content = """
    <!DOCTYPE html>
    <html>
    <body>
      <div id="widget-container"></div>
      <script type="module">
        import widget from './tnia_plotting_anywidgets.js';

        window.mockModel = {
          state: {
            annotation_mode: undefined, // Emulate the startup race condition
            sync_on_hover: true,
            axis_bounds: new Proxy({    // Emulate the AnyWidget proxy wrapper bug
              xy: { x0: 0, x1: 500, y0_js: 0, y1_js: 500, bbox: [0, 0, 500, 500] }
            }, {})
          },
          get(key) { return this.state[key]; },
          set(key, val) { this.state[key] = val; },
          save_changes() { window.syncTriggered = true; },
          on(evt, cb) {} // Mock traitlet observers
        };

        widget.render({ model: window.mockModel, el: document.getElementById('widget-container') });
      </script>
    </body>
    </html>
    """
    (base_path / "index.html").write_text(html_content)

    # 3. Serve it minimally on a random open port
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(base_path))
    httpd = http.server.HTTPServer(('localhost', 0), Handler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}"

    httpd.shutdown()
    tmpdir.cleanup()

def test_frontend_click_registers_despite_proxies_and_race_conditions(page, static_server):
    """Uses Playwright to click the JS widget and assert data flows back to the mock model."""
    page.goto(static_server)

    # Wait for JS to attach the image
    img = page.locator("img")
    img.wait_for(state="attached")

    # Force layout to match expected bounds (1x1 transparent pixel to ensure dimensions exist)
    page.evaluate("document.querySelector('img').src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs='")
    page.evaluate("document.querySelector('img').style.width = '500px'")
    page.evaluate("document.querySelector('img').style.height = '500px'")

    # 1. Resolve the simulated race condition (what happens when Python finishes syncing later)
    page.evaluate("window.mockModel.state.annotation_mode = true")

    # 2. Physically click the center of the image via headless browser
    img.click(position={"x": 250, "y": 250})

    # 3. Assert the JS properly extracted coordinates and bypassed the Proxy
    click_coords = page.evaluate("window.mockModel.state.click_coords")

    assert click_coords is not None, "Click listener failed to attach or fire."
    assert click_coords["plane"] == "xy", "Proxy traversal failed to locate the bounding box plane."
    assert 0.4 < click_coords["x"] < 0.6, "Fractional math logic via getBoundingClientRect is incorrect."
