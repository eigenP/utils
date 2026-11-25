"""Browser tests of the TNIA ortho-viewer front end (real ESM, mock anywidget model)."""

import functools
import http.server
import json
import tempfile
import threading
from pathlib import Path

import pytest

from conftest import create_sham_volume
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive


@pytest.fixture(scope="module")
def static_widget_server():
    """Serve the widget ESM plus two mock-model test harnesses on one HTTP server.

    - ``/index.html``: full range-slider and channel-control harness backed by a
      real two-channel sham volume widget.
    - ``/isolated.html``: minimal proxy and startup-race harness with realistic
      normalized ``[0, 1]`` multi-plane ``axis_bounds`` (``xy``, ``zy``, ``xz``).
    - ``/dispose.html``: the full widget with instrumented document listeners,
      ResizeObservers and timers, keeping render()'s cleanup in ``window.cleanup``.
    """
    tmpdir = tempfile.TemporaryDirectory()
    base_path = Path(tmpdir.name)

    vol = create_sham_volume()  # (2, 64, 128, 128)
    w = show_zyx_max_slice_interactive(
        [vol[0], vol[1]],
        colormap=["magenta", "green"],
        channel_labels=["Cube", "Circle"],
    )

    (base_path / "tnia_plotting_anywidgets.js").write_text(w._esm)

    full_state = {
        "image_data": w.image_data,
        "x_s": w.x_s, "y_s": w.y_s, "z_s": w.z_s,
        "x_t": w.x_t, "y_t": w.y_t, "z_t": w.z_t,
        "x_start": w.x_start, "x_end": w.x_end, "x_max_pos": w.x_max_pos, "sx": w.sx,
        "y_start": w.y_start, "y_end": w.y_end, "y_max_pos": w.y_max_pos, "sy": w.sy,
        "z_start": w.z_start, "z_end": w.z_end, "z_max_pos": w.z_max_pos, "sz": w.sz,
        "channel_names": list(w.channel_names),
        "channel_dtypes": list(w.channel_dtypes),
        "channel_colors": list(w.channel_colors),
        "vmin_list": list(w.vmin_list),
        "vmax_list": list(w.vmax_list),
        "gamma_list": list(w.gamma_list),
        "opacity_list": list(w.opacity_list),
        "histograms_data": w.histograms_data,
        "show_crosshair": w.show_crosshair,
        "sync_on_hover": w.sync_on_hover,
        "scale_bar_show": w.scale_bar_show,
        "scale_bar_fontsize_pt": w.scale_bar_fontsize_pt,
        "warning_msg": w.warning_msg,
        "save_filename": w.save_filename,
    }

    full_html = f"""<!DOCTYPE html>
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
    const state = {json.dumps(full_state)};

    window.mockModel = {{
      state,
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
    (base_path / "index.html").write_text(full_html)

    # Normalized [0, 1] multi-plane figure bounds matching the 2x2 ortho layout
    # (XY top-left, ZY top-right, XZ bottom-left, dead space bottom-right).
    isolated_bounds = {
        "xy": {"x0": 0.05, "x1": 0.65, "y0_js": 0.05, "y1_js": 0.65, "bbox": [0.05, 0.05, 0.60, 0.60]},
        "zy": {"x0": 0.70, "x1": 0.95, "y0_js": 0.05, "y1_js": 0.65, "bbox": [0.70, 0.05, 0.25, 0.60]},
        "xz": {"x0": 0.05, "x1": 0.65, "y0_js": 0.70, "y1_js": 0.95, "bbox": [0.05, 0.70, 0.60, 0.25]},
    }
    isolated_html = f"""<!DOCTYPE html>
<html>
<body style="margin: 0;">
  <div id="widget-container"></div>
  <script type="module">
    import widget from './tnia_plotting_anywidgets.js';

    const rawBounds = {json.dumps(isolated_bounds)};
    const proxiedBounds = new Proxy({{
      xy: new Proxy(rawBounds.xy, {{}}),
      zy: new Proxy(rawBounds.zy, {{}}),
      xz: new Proxy(rawBounds.xz, {{}}),
    }}, {{}});

    window.mockModel = {{
      state: {{
        annotation_mode: undefined, // Emulate the startup race condition
        sync_on_hover: true,
        axis_bounds: proxiedBounds, // Emulate the AnyWidget proxy wrapper
      }},
      get(key) {{ return this.state[key]; }},
      set(key, val) {{ this.state[key] = val; }},
      save_changes() {{ window.syncTriggered = true; }},
      on(evt, cb) {{}},
    }};

    widget.render({{ model: window.mockModel, el: document.getElementById('widget-container') }});
  </script>
</body>
</html>
"""
    (base_path / "isolated.html").write_text(isolated_html)

    # View-disposal harness: counts document keydown listeners, live
    # ResizeObservers and pending 30 s clipboard timers before the module is
    # loaded, then keeps whatever render() returns (anywidget calls it when the
    # view is removed).
    dispose_state = dict(full_state, axis_bounds=w.axis_bounds)
    dispose_html = f"""<!DOCTYPE html>
<html>
<body style="margin: 0;">
  <div id="widget-container"></div>
  <script type="module">
    const keydown = new Set();
    const addDoc = document.addEventListener.bind(document);
    const removeDoc = document.removeEventListener.bind(document);
    document.addEventListener = (type, fn, opts) => {{
      if (type === 'keydown') keydown.add(fn);
      return addDoc(type, fn, opts);
    }};
    document.removeEventListener = (type, fn, opts) => {{
      if (type === 'keydown') keydown.delete(fn);
      return removeDoc(type, fn, opts);
    }};
    window.docKeydownCount = () => keydown.size;

    let liveObservers = 0;
    const RO = window.ResizeObserver;
    window.ResizeObserver = class extends RO {{
      observe(target) {{
        if (!this._live) {{ this._live = true; liveObservers++; }}
        return super.observe(target);
      }}
      disconnect() {{
        if (this._live) {{ this._live = false; liveObservers--; }}
        return super.disconnect();
      }}
    }};
    window.liveObservers = () => liveObservers;

    const longTimers = new Set();
    const setT = window.setTimeout.bind(window);
    const clearT = window.clearTimeout.bind(window);
    window.setTimeout = (fn, ms, ...args) => {{
      const id = setT(() => {{ longTimers.delete(id); fn(...args); }}, ms);
      if (ms >= 30000) longTimers.add(id);
      return id;
    }};
    window.clearTimeout = (id) => {{ longTimers.delete(id); return clearT(id); }};
    window.pendingLongTimers = () => longTimers.size;

    const {{ default: widget }} = await import('./tnia_plotting_anywidgets.js');
    const listeners = {{}};
    window.mockModel = {{
      state: {json.dumps(dispose_state)},
      get(key) {{ return this.state[key]; }},
      set(key, val) {{
        this.state[key] = val;
        (listeners[`change:${{key}}`] || []).forEach(cb => cb());
      }},
      save_changes() {{}},
      on(evt, cb) {{ (listeners[evt] = listeners[evt] || []).push(cb); }},
    }};
    window.cleanup = widget.render({{ model: window.mockModel, el: document.getElementById('widget-container') }});
    window.rendered = true;
  </script>
</body>
</html>
"""
    (base_path / "dispose.html").write_text(dispose_html)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(base_path))
    httpd = http.server.HTTPServer(("localhost", 0), handler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    yield f"http://localhost:{port}"

    httpd.shutdown()
    httpd.server_close()
    tmpdir.cleanup()


def test_frontend_click_registers_despite_proxies_and_race_conditions(page, static_widget_server):
    """Clicks register and disambiguate planes even behind JS Proxies and late state sync.

    Strategy: mount the widget with ``annotation_mode`` initially ``undefined``
    and ``axis_bounds`` wrapped in JS ``Proxy`` objects with realistic normalized
    ``[0, 1]`` regions for ``xy``, ``zy``, and ``xz``. Resolve the startup race
    by enabling ``annotation_mode`` after mount, then click the center of each
    panel and the dead bottom-right corner.
    Expected outcome: each panel click reports its own plane name and accurate
    fractional coordinates; clicking outside all three panels leaves
    ``click_coords`` untouched.
    """
    page.goto(f"{static_widget_server}/isolated.html")

    img = page.locator("img")
    img.wait_for(state="attached")

    # Give the image a deterministic 500x500 box via a 1x1 transparent GIF.
    page.evaluate("""() => {
        const img = document.querySelector('img');
        img.src = 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
        img.style.width = '500px';
        img.style.height = '500px';
    }""")
    page.wait_for_function(
        "() => { const r = document.querySelector('img').getBoundingClientRect(); "
        "return r.width === 500 && r.height === 500; }"
    )

    # Resolve the simulated startup race (Python finishes syncing after mount).
    page.evaluate("window.mockModel.state.annotation_mode = true")

    for plane, fx, fy in (
        ("xy", 0.35, 0.35),
        ("zy", 0.825, 0.35),
        ("xz", 0.35, 0.825),
    ):
        page.evaluate("window.mockModel.state.click_coords = null")
        img.click(position={"x": fx * 500, "y": fy * 500})
        click_coords = page.evaluate("window.mockModel.state.click_coords")
        assert click_coords is not None, f"Click listener failed to fire on {plane}."
        assert click_coords["plane"] == plane, f"Expected plane {plane!r}, got {click_coords}"
        assert click_coords["x"] == pytest.approx(fx, abs=0.02)
        assert click_coords["y"] == pytest.approx(fy, abs=0.02)

    # Negative control: clicking the dead bottom-right corner matches no plane.
    page.evaluate("window.mockModel.state.click_coords = null")
    img.click(position={"x": 0.85 * 500, "y": 0.85 * 500})
    assert page.evaluate("window.mockModel.state.click_coords") is None


def test_frontend_rangeslider_ui(page, static_widget_server):
    """Dragging the middle range-slider thumb translates the slab while preserving its width.

    Strategy: locate the ``Translate Range`` thumb on the X range slider, drag it
    30 px to the right, and read ``x_start`` and ``x_end`` back from the mock model.
    Expected outcome: both bounds shift upward by the exact same delta.
    """
    page.goto(static_widget_server)

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


def test_frontend_rangeslider_integer_step_on_both_inputs(page, static_widget_server):
    """Both numeric boxes of a range slider must expose an integer step when unscaled.

    The fixture volume is built without `pixel_sizes`, so every scale trait is
    1.0 and the boxes hold raw voxel indices. Strategy: read the `step`
    attribute the widget assigned to each number input. A start box stepping by
    1 while its end box steps by 0.01 (or carries no step at all) lets the
    spinner arrows produce fractional slab bounds on one side only, so every
    box is expected to report a step of exactly "1".
    """
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")

    steps = page.evaluate(
        "() => [...document.querySelectorAll('input[type=number]')].map(i => i.step)"
    )

    # Three sliders (X, Y, Z), each with a start and an end box.
    assert len(steps) == 6
    assert set(steps) == {"1"}


def test_frontend_channel_rebuild_does_not_leak_listeners(page, static_widget_server):
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
    page.goto(static_widget_server)
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


def test_frontend_channel_inputs_allow_negative_signed_limits_and_positive_gamma(page, static_widget_server):
    """Signed-integer channels accept a negative vmin, unsigned ones clamp at 0, and gamma stays positive.

    Strategy: rebuild the channel rows with ``['int16', 'uint8']`` dtypes, enter
    ``-300`` into both ``vmin`` inputs and ``0`` into ``gamma``.
    Expected outcome: the signed channel keeps ``-300``, the unsigned channel
    clamps to ``0``, and ``gamma`` clamps to ``0.05`` so it never floods the panel.
    """
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")

    # Rebuild the channel rows as an int16 and a uint8 channel
    page.evaluate("window.mockModel.state.channel_dtypes = ['int16', 'uint8']")
    page.evaluate("window.mockModel.set('channel_names', ['Signed', 'Unsigned'])")

    def type_into(label, index, text):
        page.evaluate(
            """([label, index, text]) => {
                const inp = [...document.querySelectorAll('span')]
                    .filter(s => s.textContent === label)[index].nextElementSibling;
                inp.value = text;
                inp.dispatchEvent(new Event('change'));
            }""",
            [label, index, text],
        )

    type_into("vmin", 0, "-300")
    type_into("vmin", 1, "-300")
    type_into("gamma", 0, "0")
    assert page.evaluate("window.mockModel.get('vmin_list')") == [-300, 0]
    assert page.evaluate("window.mockModel.get('gamma_list')")[0] == 0.05


def test_frontend_histogram_vrange_slider_and_width(page, static_widget_server):
    """The vmin/vmax slider under each histogram drives the model; the histogram fills the row.

    The fixture channels are uint8 (histogram range 0..255). Strategy: drag the
    vmax thumb of channel 0 to the middle of its track; mid-drag the thumb must
    follow the pointer while the model is untouched (no re-render per pointer
    move); after release vmax must be an integer near 128 and vmin untouched;
    drag the middle thumb and check vmin and vmax shift together with the window
    width preserved. Then compare the histogram width in a wide and a narrow
    viewport: it must grow beyond its 160 px minimum when space is available and
    never shrink below it, and the canvas buffer must follow the laid-out width.
    """
    page.set_viewport_size({"width": 1800, "height": 900})
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")

    track = page.locator("div[title='vmax']").first.locator("..")
    tbox = track.bounding_box()
    vmax_thumb = page.locator("div[title='vmax']").first
    b = vmax_thumb.bounding_box()
    vmax_before = page.evaluate("window.mockModel.get('vmax_list')[0]")
    page.evaluate("window.saveTriggered = false")
    page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    page.mouse.down()
    page.mouse.move(tbox["x"] + tbox["width"] / 2, b["y"] + b["height"] / 2, steps=5)
    # While dragging only the slider moves; the model (and the Python render)
    # is updated once, on release.
    assert page.evaluate("window.mockModel.get('vmax_list')[0]") == vmax_before
    assert page.evaluate("window.saveTriggered") is False
    moved = vmax_thumb.bounding_box()
    assert abs(moved["x"] + moved["width"] / 2 - (tbox["x"] + tbox["width"] / 2)) < 4
    page.mouse.up()
    assert page.evaluate("window.saveTriggered") is True

    vmin0 = page.evaluate("window.mockModel.get('vmin_list')[0]")
    vmax0 = page.evaluate("window.mockModel.get('vmax_list')[0]")
    assert isinstance(vmax0, int) and abs(vmax0 - 128) <= 3
    assert vmin0 == 0

    mid = page.locator("div[title='Shift vmin and vmax']").first.bounding_box()
    page.mouse.move(mid["x"] + mid["width"] / 2, mid["y"] + mid["height"] / 2)
    page.mouse.down()
    page.mouse.move(mid["x"] + mid["width"] / 2 + tbox["width"] / 8, mid["y"] + mid["height"] / 2)
    page.mouse.up()
    vmin1 = page.evaluate("window.mockModel.get('vmin_list')[0]")
    vmax1 = page.evaluate("window.mockModel.get('vmax_list')[0]")
    assert vmin1 > 0 and vmax1 - vmin1 == vmax0 - vmin0

    canvas_w = "() => document.querySelector('canvas').getBoundingClientRect().width"
    wide = page.evaluate(canvas_w)
    page.set_viewport_size({"width": 700, "height": 900})
    page.wait_for_function(
        "wide => { const c = document.querySelector('canvas'); "
        "const w = c.getBoundingClientRect().width; "
        "return w < wide && Math.abs(c.width - w) <= 2; }",
        arg=wide,
    )
    narrow = page.evaluate(canvas_w)
    assert wide > 200
    assert narrow >= 160 and narrow < wide
    # The drawing buffer follows the laid-out width (no blurry stretching).
    assert abs(page.evaluate("() => document.querySelector('canvas').width") - narrow) <= 2


def test_frontend_scale_bar_checkbox_toggles_model(page, static_widget_server):
    """The "Scale Bar" checkbox writes `scale_bar_show` and follows model changes.

    Strategy: the checkbox starts checked (default on), unchecking it must set
    the trait to false and save, and a change pushed from Python must be
    reflected back in the checkbox.
    """
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")

    cb = page.get_by_label("Scale Bar")
    assert cb.is_checked()
    page.evaluate("window.saveTriggered = false")
    cb.uncheck()
    assert page.evaluate("window.mockModel.get('scale_bar_show')") is False
    assert page.evaluate("window.saveTriggered") is True

    page.evaluate("window.mockModel.set('scale_bar_show', true)")
    assert cb.is_checked()


def test_frontend_scale_bar_fontsize_box(page, static_widget_server):
    """The scale-bar size box writes `scale_bar_fontsize_pt`: a number, 0 (label hidden) or null (auto).

    Strategy: type 14, then 0, then clear the box, and read the trait back;
    a negative entry is clamped to 0 and non-numeric text is rejected.
    """
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")
    box = page.locator("input[placeholder='auto']")

    def enter(text):
        box.fill(text)
        box.dispatch_event("change")
        return page.evaluate("window.mockModel.get('scale_bar_fontsize_pt')")

    assert enter("14") == 14
    assert enter("0") == 0
    assert enter("-3") == 0
    assert enter("") is None
    assert enter("abc") is None and box.input_value() == ""


def test_frontend_copy_png_button_writes_clipboard(page, static_widget_server):
    """The "📋 PNG" button asks Python for the PNG and writes it to the clipboard.

    Strategy: clicking must bump `copy_png_trigger`; pushing a PNG into
    `copy_png_data` (what `_copy_png` does) must resolve the pending
    ClipboardItem, and reading the clipboard back must return those bytes.
    Negative control: an "error" payload must not write and must show ❌.
    """
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")
    btn = page.get_by_role("button", name="📋 PNG", exact=True)
    # The kwargs copy button is labelled separately so the two are not confused.
    page.get_by_role("button", name="📋 kwargs", exact=True).wait_for(state="visible")

    before = page.evaluate("window.mockModel.get('copy_png_trigger') || 0")
    btn.click()
    assert page.evaluate("window.mockModel.get('copy_png_trigger')") == before + 1

    png_b64 = page.evaluate("window.mockModel.get('image_data')")
    page.evaluate("data => window.mockModel.set('copy_png_data', data)", png_b64)
    page.get_by_role("button", name="✅").wait_for(state="visible", timeout=5000)
    roundtrip = page.evaluate("""async () => {
        const items = await navigator.clipboard.read();
        const blob = await items[0].getType('image/png');
        const bytes = new Uint8Array(await blob.arrayBuffer());
        let s = ''; for (const b of bytes) s += String.fromCharCode(b);
        return btoa(s);
    }""")
    assert roundtrip[:16] == png_b64[:16]  # PNG header survives

    btn = page.get_by_role("button", name="📋 PNG")
    btn.wait_for(state="visible", timeout=5000)
    btn.click()
    page.evaluate("window.mockModel.set('copy_png_data', 'error')")
    page.get_by_role("button", name="❌").wait_for(state="visible", timeout=5000)


def _type_into_channel_box(page, label, index, text):
    """Type `text` into the `label` box (vmin, vmax, gamma, opacity) of channel `index` and commit it."""
    page.evaluate(
        """([label, index, text]) => {
            const inp = [...document.querySelectorAll('span')]
                .filter(s => s.textContent === label)[index].nextElementSibling;
            inp.value = text;
            inp.dispatchEvent(new Event('change'));
        }""",
        [label, index, text],
    )


def test_frontend_vmin_vmax_boxes_keep_vmin_at_most_vmax(page, static_widget_server):
    """A vmin typed above vmax (or a vmax below vmin) is clamped to the other bound.

    The histogram slider already keeps vmin <= vmax, but the text boxes only
    clamped to the dtype range, so typing vmin above vmax sent an inverted
    pair (which froze the single-channel render). Strategy: set channel 0 to
    vmin 10 / vmax 100, type vmin 200, then vmax 20.
    Expected outcome: vmin becomes 100 (the current vmax), then vmax becomes
    100 (the current vmin), each shown in its box.
    Negative controls: a value inside the range is kept, and an empty (auto)
    vmax does not clamp vmin.
    """
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")
    page.evaluate("window.mockModel.set('vmin_list', [10, 0]); window.mockModel.set('vmax_list', [100, 255])")

    _type_into_channel_box(page, "vmin", 0, "200")
    assert page.evaluate("window.mockModel.get('vmin_list')") == [100, 0]
    _type_into_channel_box(page, "vmin", 0, "50")
    assert page.evaluate("window.mockModel.get('vmin_list')") == [50, 0]
    _type_into_channel_box(page, "vmax", 0, "20")
    assert page.evaluate("window.mockModel.get('vmax_list')") == [50, 255]
    box_values = page.evaluate(
        "() => [...document.querySelectorAll('span')]"
        ".filter(s => s.textContent === 'vmin' || s.textContent === 'vmax')"
        ".slice(0, 2).map(s => s.nextElementSibling.value)"
    )
    assert box_values == ["50", "50"]

    _type_into_channel_box(page, "vmax", 0, "")
    _type_into_channel_box(page, "vmin", 0, "240")
    assert page.evaluate("window.mockModel.get('vmin_list')") == [240, 0]
    assert page.evaluate("window.mockModel.get('vmax_list')") == ["", 255]


def test_frontend_range_slider_with_zero_max_is_hidden_and_stays_at_zero(page, static_widget_server):
    """A dimension of size 1 (Z of a 2D image, max position 0) hides its range row and never sends 1.

    ``model.get(maxTrait) || 1`` turned the 0 into 1, so the Z track spanned
    [0, 1] and the end box or thumb could send z_end = 1, out of range.
    Strategy: set ``z_max_pos`` (and the Z slab) to 0, as Python does for 2D
    input, then type 1 into the Z end box.
    Expected outcome: the Z row is hidden and z_end stays 0.
    Negative control: the X and Y rows stay visible.
    """
    page.goto(static_widget_server)
    page.get_by_text("Z Range").wait_for(state="visible")
    page.evaluate(
        "() => { const m = window.mockModel;"
        " m.set('z_start', 0); m.set('z_end', 0); m.set('z_max_pos', 0); }"
    )
    assert not page.get_by_text("Z Range").is_visible()
    assert page.get_by_text("X Range").is_visible() and page.get_by_text("Y Range").is_visible()

    page.evaluate(
        "() => { const z_end = [...document.querySelectorAll('input[type=number]')][5];"
        " z_end.value = '1'; z_end.dispatchEvent(new Event('change')); }"
    )
    assert page.evaluate("window.mockModel.get('z_end')") == 0


def test_frontend_copy_kwargs_writes_the_clipboard_from_the_click(page, static_widget_server):
    """The "📋 kwargs" button starts its clipboard write inside the click and copies on every click.

    The kwargs were written only when ``copy_params_string`` changed, after the
    Python round trip: outside the click's user activation (Safari rejects
    such writes) and never for a second click on an unchanged view; failures
    only reached the console. Strategy: count ``navigator.clipboard.write``
    calls; click, then answer as Python does (clear, then the string); click
    again with the same string; finally make the write fail.
    Expected outcome: each click starts one write before Python answers, each
    answer lands in the clipboard with a ✅, and a failed write shows ❌.
    """
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(static_widget_server)
    page.get_by_text("X Range").wait_for(state="visible")
    page.evaluate("""() => {
        window.clipWrites = 0;
        const write = navigator.clipboard.write.bind(navigator.clipboard);
        navigator.clipboard.write = (items) => { window.clipWrites++; return write(items); };
    }""")
    kwargs = "vmin=[0.0],\nvmax=[255.0],"

    for n in (1, 2):
        btn = page.get_by_role("button", name="📋 kwargs", exact=True)
        btn.wait_for(state="visible", timeout=5000)
        btn.click()
        assert page.evaluate("window.clipWrites") == n  # started inside the click
        assert page.evaluate("window.mockModel.get('copy_params_trigger')") == n
        page.evaluate("s => { window.mockModel.set('copy_params_string', ''); "
                      "window.mockModel.set('copy_params_string', s); }", kwargs)
        page.get_by_role("button", name="✅").wait_for(state="visible", timeout=5000)
        assert page.evaluate("navigator.clipboard.readText()") == kwargs

    page.evaluate("() => { navigator.clipboard.write = () => Promise.reject(new Error('denied')); }")
    btn = page.get_by_role("button", name="📋 kwargs", exact=True)
    btn.wait_for(state="visible", timeout=5000)
    btn.click()
    page.evaluate("s => window.mockModel.set('copy_params_string', s)", kwargs + "\n")
    page.get_by_role("button", name="❌").wait_for(state="visible", timeout=5000)


def test_frontend_view_cleanup_releases_document_listener_and_observers(page, static_widget_server):
    """render() returns a cleanup that removes everything the view registered outside its element.

    anywidget disposes a view by removing its model listeners and then calling
    the cleanup render() returned; a 'destroy' handler never runs then. The
    view used to return nothing, so each re-run cell left a document keydown
    listener (holding the model and image) that still sent ``hover_coords``
    when 'c' was typed, plus its histogram ResizeObservers. Strategy: in the
    instrumented harness, hover the XY panel and press 'c' (positive control:
    ``hover_coords`` is sent), click the PNG and the kwargs buttons twice each
    before Python answers, then call the cleanup and press 'c' again.
    Expected outcome: a second click supersedes the first and drops its 30 s
    timer (one pending timer per button, so the cleanup can clear it, and the
    stale timer cannot later reset the newer pending copy); the cleanup is a
    function; afterwards no document keydown listener, live ResizeObserver or
    pending 30 s copy timer is left, and 'c' sends nothing.
    """
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    page.goto(f"{static_widget_server}/dispose.html")
    page.wait_for_function("window.rendered === true")
    page.locator("img").wait_for(state="visible")

    assert page.evaluate("window.docKeydownCount()") == 1
    assert page.evaluate("window.liveObservers()") == 2  # one per channel histogram

    box = page.locator("img").bounding_box()
    x0, y0, bw, bh = page.evaluate("window.mockModel.get('axis_bounds').xy.bbox")
    page.mouse.move(box["x"] + (x0 + bw / 2) * box["width"], box["y"] + (y0 + bh / 2) * box["height"])
    page.keyboard.press("c")
    assert page.evaluate("window.mockModel.get('hover_coords')")["plane"] == "xy"

    for _ in range(2):  # the second click supersedes the first
        page.get_by_role("button", name="📋 PNG", exact=True).click()
        page.get_by_role("button", name="📋 kwargs", exact=True).click()
    assert page.evaluate("window.pendingLongTimers()") == 2

    assert page.evaluate("typeof window.cleanup") == "function"
    page.evaluate("window.cleanup()")
    assert page.evaluate("window.docKeydownCount()") == 0
    assert page.evaluate("window.liveObservers()") == 0
    assert page.evaluate("window.pendingLongTimers()") == 0

    page.evaluate("window.mockModel.state.hover_coords = null")
    page.keyboard.press("c")
    assert page.evaluate("window.mockModel.get('hover_coords')") is None
