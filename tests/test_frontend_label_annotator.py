"""Browser tests of the LabelAnnotator front end (real ESM, mock anywidget model).

The widget's initial state comes from a real Python ``LabelAnnotator``; label
messages the browser sends back are decoded with the Python wire decoder and
applied to the Python widget, so both halves of the protocol are exercised.
"""

import base64
import functools
import http.server
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from eigenp_utils.label_annotator import LabelAnnotator, decode_label_slice

H, W = 40, 40
VIEW_PX = 400  # viewport width -> zoom 10 (one image pixel = 10 CSS px)

_HARNESS = """<!DOCTYPE html>
<html><head><link rel="stylesheet" href="label_annotator.css"></head><body style="margin:0">
<div id="c" style="width:%(view)dpx"></div>
<script type="module">
import widget from './label_annotator.js';
const init = %(state)s;
function b64ToDataView(s) {
  const bin = atob(s); const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new DataView(u.buffer);
}
const state = {};
for (const [k, v] of Object.entries(init)) state[k] = (v && v.__b64__ !== undefined) ? b64ToDataView(v.__b64__) : v;
const listeners = {};
window.sent = [];
window.mockModel = {
  get(k) { return state[k]; },
  set(k, v) { state[k] = v; (listeners['change:' + k] || []).forEach((f) => f()); },
  on(evt, cb) { (listeners[evt] = listeners[evt] || []).push(cb); },
  off(evt, cb) { listeners[evt] = (listeners[evt] || []).filter((f) => f !== cb); },
  save_changes() { window.sent.push(Object.keys(state).length); },
};
window.lastLabelsB64 = () => {
  const v = state.labels_from_js;
  if (!v) return null;
  const u = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  let s = ''; for (const b of u) s += String.fromCharCode(b);
  return btoa(s);
};
widget.render({ model: window.mockModel, el: document.getElementById('c') });
window.ready = true;
</script></body></html>
"""


def _state_json(w: LabelAnnotator) -> str:
    out = {}
    for name in w.trait_names(sync=True):
        if name.startswith("_") or name in ("layout", "comm", "keys", "log", "tabbable", "tooltip"):
            continue
        v = getattr(w, name)
        if isinstance(v, (bytes, bytearray, memoryview)):
            out[name] = {"__b64__": base64.b64encode(bytes(v)).decode()}
        elif v is None or isinstance(v, (bool, int, float, str, list, dict)):
            out[name] = v
    return json.dumps(out)


@pytest.fixture
def serve():
    servers = []

    def _serve(widget: LabelAnnotator, css: bool = True) -> str:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        (base / "label_annotator.js").write_text(LabelAnnotator._esm)
        (base / "label_annotator.css").write_text(LabelAnnotator._css if css else "")
        (base / "index.html").write_text(_HARNESS % {"view": VIEW_PX, "state": _state_json(widget)})
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(base))
        httpd = http.server.HTTPServer(("localhost", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append((httpd, tmp))
        return f"http://localhost:{httpd.server_address[1]}/index.html"

    yield _serve
    for httpd, tmp in servers:
        httpd.shutdown()
        httpd.server_close()
        tmp.cleanup()


def _open(page, url):
    page.goto(url)
    page.wait_for_function("window.ready === true")
    page.wait_for_function(
        f"""() => {{
            const c = document.querySelector('canvas.lbl-img');
            const vp = document.querySelector('.lbl-viewport');
            const st = document.querySelector('.lbl-stage');
            if (!c || !vp || !st || c.width !== {W}) return false;
            const cw = c.getBoundingClientRect().width;
            return vp.clientWidth > 0 && st.style.transform.includes('scale(')
                && Math.abs(cw - vp.clientWidth) <= 1;
        }}"""
    )


def _client_xy(page, x, y):
    r = page.evaluate("(() => { const b = document.querySelector('canvas.lbl-img').getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })()")
    return r[0] + (x + 0.5) * r[2] / W, r[1] + (y + 0.5) * r[3] / H


def _stroke(page, pts):
    page.mouse.move(*_client_xy(page, *pts[0]))
    page.mouse.down()
    for p in pts[1:]:
        page.mouse.move(*_client_xy(page, *p), steps=4)
    page.mouse.up()


def _last_labels(page):
    b64 = page.evaluate("window.lastLabelsB64()")
    assert b64 is not None, "front end sent no labels"
    return decode_label_slice(base64.b64decode(b64))


def _widget(**kw):
    img = np.random.default_rng(0).random((H, W)).astype(np.float32)
    return LabelAnnotator(img, brush_radius=0, **kw)


def test_closed_stroke_is_filled_and_python_receives_it(page, serve):
    """Closed brush strokes auto-fill their interior and round-trip to Python.

    Strategy: draw a closed square perimeter from (10, 10) to (30, 30) with
    ``brush_radius=0`` in instance mode (``auto_fill=True``), decode the wire
    payload emitted by JS, and feed it into ``w.labels_from_js``.
    Expected outcome: the decoded mask equals the exact filled ``[10..30]^2``
    square with exterior pixels at 0, Python updates ``w.labels`` identically,
    and ``w.active_label`` advances to 2.
    """
    w = _widget()
    _open(page, serve(w))
    square = [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]
    _stroke(page, square)
    z, lab = _last_labels(page)
    assert z == 0
    assert lab[20, 20] == 1, "interior of a closed stroke is filled (auto_fill)"
    assert lab[10, 20] == 1 and lab[30, 30] == 1, "stroke pixels are labelled"
    assert lab[5, 5] == 0 and lab[35, 35] == 0, "exterior stays unlabelled"
    # closed-form oracle: the filled square with radius-0 brush is exactly [10..30]^2
    expected = np.zeros((H, W), np.uint16)
    expected[10:31, 10:31] = 1
    np.testing.assert_array_equal(lab, expected)
    # Python side applies the message and advances the instance id
    w.labels_from_js = base64.b64decode(page.evaluate("window.lastLabelsB64()"))
    np.testing.assert_array_equal(w.labels, expected)
    assert w.active_label == 2


def test_painting_coordinates_do_not_depend_on_host_css(page, serve):
    """Hosts that fail to inject the stylesheet must not shift the paint position.

    Strategy: serve the widget with an empty CSS file (``css=False``) and tap a
    single pixel ``(7, 12)`` with ``brush_radius=0`` and ``auto_fill=False``.
    Expected outcome: only pixel ``[12, 7]`` is labelled with id 1.
    """
    w = _widget()
    w.auto_fill = False
    _open(page, serve(w, css=False))
    _stroke(page, [(7, 12), (7, 12)])
    _, lab = _last_labels(page)
    assert lab[12, 7] == 1 and lab.sum() == 1


def test_open_stroke_without_fill_and_eraser_and_undo(page, serve):
    """Open strokes, the eraser hotkey, and Ctrl+Z undo behave consistently.

    Strategy: disable ``auto_fill``, paint a horizontal line across ``y=20``,
    press ``e`` to select the eraser and cross the line vertically at ``x=20``,
    then press ``Control+z`` to undo the erasure.
    Expected outcome: the initial stroke paints 31 pixels, erasing removes only
    the single intersection ``(20, 20)``, and undo restores all 31 pixels.
    """
    w = _widget()
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(5, 20), (35, 20)])
    _, lab = _last_labels(page)
    assert lab[20, 5:36].tolist() == [1] * 31 and lab.sum() == 31
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("e")
    _stroke(page, [(20, 15), (20, 25)])
    _, lab = _last_labels(page)
    assert lab[20, 20] == 0 and lab.sum() == 30, "eraser removes exactly the crossed pixel"
    page.keyboard.press("Control+z")
    _, lab = _last_labels(page)
    assert lab[20, 20] == 1 and lab.sum() == 31, "undo restores the erased pixel"


def test_no_overwrite_protects_existing_labels(page, serve):
    """With ``overwrite=False`` (default), painting across an existing label preserves it.

    Strategy: pre-populate column 20 with label 7 and paint a horizontal stroke
    crossing column 20 at row 10.
    Expected outcome: ``lab[10, 20]`` remains 7 while unlabelled pixels along
    the stroke receive ``w.active_label``.
    """
    init = np.zeros((H, W), np.uint32)
    init[:, 20] = 7
    w = _widget(labels=init)
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(5, 10), (35, 10)])  # crosses column 20 at row 10
    _, lab = _last_labels(page)
    assert lab[10, 20] == 7, "overwrite=False keeps other labels"
    assert lab[10, 5] == w.active_label


def test_semantic_fill_tool_and_class_hotkeys(page, serve):
    """Semantic class hotkeys switch ``active_label`` and ``f`` triggers flood fill.

    Strategy: open a semantic annotator with two classes, press ``2`` to select
    class 2, press ``f`` for flood fill, and click inside the empty canvas.
    Expected outcome: ``active_label`` becomes 2 and the entire slice is filled
    with class id 2.
    """
    w = LabelAnnotator(np.zeros((H, W), np.float32), mode="semantic", classes=["mito", "perox"], brush_radius=0)
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("2")
    assert page.evaluate("window.mockModel.get('active_label')") == 2
    page.keyboard.press("f")
    page.mouse.click(*_client_xy(page, 3, 3))
    _, lab = _last_labels(page)
    assert (lab == 2).all(), "flood fill of an empty slice labels every pixel"
