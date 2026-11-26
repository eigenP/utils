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

from eigenp_utils.label_annotator import LabelAnnotator, decode_label_slice, encode_label_slice

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
window.widget = widget;
window.b64ToDataView = b64ToDataView;
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
        (base / "index.html").write_text(_HARNESS % {"view": VIEW_PX + (230 if css else 0), "state": _state_json(widget)})
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


# ---------------------------------------------------------------- z-linking
def _widget3d(**kw):
    img = np.random.default_rng(1).random((3, H, W)).astype(np.float32)
    lab = np.zeros((3, H, W), np.uint32)
    lab[0, 5:15, 5:15] = 4  # z=0 only
    lab[1, 25:35, 25:35] = 6
    return LabelAnnotator(img, axes="ZYX", labels=lab, brush_radius=0, **kw), lab


def _get(page, name):
    return page.evaluate(f"window.mockModel.get('{name}')")


def test_alt_arrows_step_z_and_keep_label_toggle(page, serve):
    """Alt+ArrowRight/Left step z by +1/-1 (like ArrowUp/Down); 'keep label' sets ``auto_increment`` False."""
    w, _ = _widget3d()
    _open(page, serve(w))
    assert _get(page, "z") == 1
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Alt+ArrowRight")
    assert _get(page, "z") == 2
    page.keyboard.press("Alt+ArrowRight")  # clamped at the last slice
    assert _get(page, "z") == 2
    page.keyboard.press("Alt+ArrowLeft")
    page.keyboard.press("Alt+ArrowLeft")
    assert _get(page, "z") == 0
    assert _get(page, "auto_increment") is True
    page.locator("input.lbl-keep_label").check()
    assert _get(page, "auto_increment") is False


def test_merge_tool_sends_request_and_undo_routes_to_python(page, serve):
    """The merge tool asks Python to merge the clicked id into the active id; undo then targets that op.

    Strategy: active id 6, press ``m`` and click the id-4 mask (on slice 0);
    apply the request in Python, mirror ``track_undo_ids`` into the page and
    press ctrl+z.
    Expected outcome: the request is ``{op: merge, src: 4, dst: 6}``; Python
    relabels 4 -> 6; ctrl+z sends ``{op: undo, serial: <that op>}``, which
    restores id 4 in Python.
    """
    w, lab = _widget3d()
    w.z = 0
    w.active_label = 6
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("m")
    assert _get(page, "tool") == "merge"
    page.mouse.click(*_client_xy(page, 10, 10))
    req = _get(page, "track_request")
    assert {k: req[k] for k in ("op", "src", "dst")} == {"op": "merge", "src": 4, "dst": 6}
    w.track_request = req
    assert not (w.labels == 4).any()
    page.evaluate(f"window.mockModel.set('track_undo_ids', {w.track_undo_ids})")
    page.keyboard.press("Control+z")
    req = _get(page, "track_request")
    assert req["op"] == "undo" and req["serial"] == w.track_undo_ids[-1]
    w.track_request = req
    np.testing.assert_array_equal(w.labels, lab)


def test_adjacent_outlines_drawn_only_on_mask_edges(page, serve):
    """With ``show_adjacent`` the id-4 mask of slice z-1 appears as a (dotted) outline on slice z."""
    w, _ = _widget3d(show_adjacent=True)
    assert w.z == 1
    _open(page, serve(w))
    alpha = page.evaluate(
        """() => { const c = document.querySelector('canvas.lbl-adj');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        const a = []; for (let i = 3; i < d.length; i += 4) a.push(d[i]); return a; }"""
    )
    a = np.asarray(alpha).reshape(H, W)
    edge = np.zeros((H, W), bool)
    edge[5:15, 5:15] = True
    edge[6:14, 6:14] = False
    assert a[~edge].max() == 0, "only the outline of the neighbour's mask is drawn"
    assert (a[edge] > 0).sum() >= edge.sum() // 2 - 1, "dotted outline covers about half the edge pixels"
    page.locator("input.lbl-show_adjacent").uncheck()
    assert _get(page, "show_adjacent") is False


def test_modified_letter_keys_do_not_switch_tools(page, serve):
    """ctrl+s / ctrl+f / ctrl+p (host shortcuts) must not arm split / fill / picker; plain keys still do."""
    w, _ = _widget3d()
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    for key in ("Control+s", "Control+f", "Control+p", "Meta+m", "Control+n"):
        page.keyboard.press(key)
        assert _get(page, "tool") == "brush", key
    assert _get(page, "new_label_trigger") == 0
    page.keyboard.press("s")
    assert _get(page, "tool") == "split"


def test_evicted_track_marker_is_not_readded_above_later_strokes(page, serve):
    """A track-op marker pushed off the bounded undo stack must not come back on top.

    Strategy: Python reports track op 1; then more single-pixel strokes than
    the undo stack holds evict its marker; then Python reports op 2. Press
    ctrl+z twice.
    Expected outcome: the first undo asks Python to undo op 2; the second undoes
    the newest stroke (a slice undo), not op 1 (which predates every stroke).
    """
    w, _ = _widget3d()
    w.auto_fill = False
    w.auto_increment = False
    _open(page, serve(w))
    page.evaluate("window.mockModel.set('track_undo_ids', [1])")
    for i in range(32):  # UNDO_MAX = 30
        _stroke(page, [(i + 2, 2), (i + 2, 2)])
    page.evaluate("window.mockModel.set('track_undo_ids', [1, 2])")
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Control+z")
    req = _get(page, "track_request")
    assert req["op"] == "undo" and req["serial"] == 2
    page.evaluate("window.mockModel.set('track_undo_ids', [1])")  # Python undid op 2
    page.keyboard.press("Control+z")
    assert _get(page, "track_request")["n"] == req["n"], "second undo must not go to Python"
    _, lab = _last_labels(page)
    assert lab[2, 33] == 0 and lab[2, 32] != 0, "the newest stroke was undone"


def test_track_request_counter_continues_from_model(page, serve):
    """A (re-)rendered view numbers requests after the model's last one, so identical requests still change the trait."""
    w, _ = _widget3d()
    w.z = 0
    w.active_label = 6
    w.track_request = {"op": "noop", "n": 7}  # unknown op: ignored by Python
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("m")
    page.mouse.click(*_client_xy(page, 10, 10))
    assert _get(page, "track_request")["n"] == 8


def test_labels_rev_reloads_slice_with_unchanged_bytes(page, serve):
    """A ``labels_rev`` change reloads ``labels_to_js`` even if its bytes did not change.

    Strategy: paint a stroke (local only), then bump ``labels_rev`` without
    changing ``labels_to_js`` (Python rewrote the slice to exactly what it last
    sent), then paint a second stroke elsewhere.
    Expected outcome: the second message holds the second stroke but not the
    first (the browser reloaded Python's slice instead of keeping stale pixels).
    """
    w, _ = _widget3d()
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(2, 2), (8, 2)])
    _, lab = _last_labels(page)
    assert lab[2, 2:9].all()
    page.evaluate("window.mockModel.set('labels_rev', 1)")
    _stroke(page, [(2, 38), (8, 38)])
    _, lab = _last_labels(page)
    assert lab[38, 2:9].all() and not lab[2, 2:9].any()


def test_label_input_is_clamped_to_uint16(page, serve):
    """Typing an id above 65535 sets 65535 (uint16 painting would wrap larger ids)."""
    w, _ = _widget3d()
    _open(page, serve(w))
    page.locator("input.lbl-label").fill("70000")
    page.locator("input.lbl-label").dispatch_event("change")
    assert _get(page, "active_label") == 65535


# ------------------------------------------------- sparse annotation: background, delete
def test_background_brush_paints_open_scribbles_with_reserved_id(page, serve):
    """The background brush (``g``) paints 65535 as an open scribble, never filled, in a hatched grey.

    Strategy: ``background=True``; press ``g`` and draw the same closed square
    that the cell brush would fill; read the label canvas.
    Expected outcome: the stroke pixels are 65535 and the interior stays 0
    (no fill, although ``auto_fill`` is on); Python applies the message
    without advancing the active id; background pixels are drawn grey
    (r == g == b) and opaque enough to see; ``g`` does nothing without
    ``background``.
    """
    w = _widget(background=True)
    assert w.auto_fill is True
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("g")
    assert _get(page, "tool") == "background"
    square = [(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]
    _stroke(page, square)
    _, lab = _last_labels(page)
    assert lab[10, 20] == 65535 and lab[30, 30] == 65535
    assert lab[20, 20] == 0, "background strokes are not closed / filled"
    assert set(np.unique(lab).tolist()) == {0, 65535}
    active = w.active_label
    w.labels_from_js = base64.b64decode(page.evaluate("window.lastLabelsB64()"))
    assert w.active_label == active
    assert w.counts()["background_px"] == int((lab == 65535).sum())
    rgba = page.evaluate(
        """() => { const c = document.querySelector('canvas.lbl-lab');
        return Array.from(c.getContext('2d').getImageData(20, 10, 1, 1).data); }"""
    )
    assert rgba[0] == rgba[1] == rgba[2] and rgba[3] >= 140, rgba
    page.locator("input.lbl-display").check()
    rgba = page.evaluate(
        """() => { const c = document.querySelector('canvas.lbl-lab');
        return Array.from(c.getContext('2d').getImageData(20, 10, 1, 1).data); }"""
    )
    assert rgba[3] > 0, "background stays visible in outline mode"
    assert page.locator('button[data-tool="background"]').is_visible()

    w2 = _widget()
    _open(page, serve(w2))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("g")
    assert _get(page, "tool") == "brush"
    assert not page.locator('button[data-tool="background"]').is_visible()


def test_delete_tool_requests_slice_or_all_slices_and_undo_routes_to_python(page, serve):
    """``x`` arms delete: click -> delete on this slice, shift+click -> all slices; ctrl+z undoes in Python.

    Strategy: 3-slice volume (id 4 on z=0); on z=0 press ``x``, click the
    mask, apply the request in Python; mirror ``track_undo_ids``; ctrl+z;
    then shift+click the mask.
    Expected outcome: the first request is ``{op: delete, id: 4, z: 0}`` and
    removes id 4; ctrl+z sends the undo for that serial, which restores it;
    shift+click sends ``z: null`` (all slices). Clicking unannotated pixels
    sends nothing.
    """
    w, lab = _widget3d()
    w.z = 0
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("x")
    assert _get(page, "tool") == "delete"
    page.mouse.click(*_client_xy(page, 30, 30))  # unannotated on z=0
    assert not _get(page, "track_request")
    page.mouse.click(*_client_xy(page, 10, 10))
    req = _get(page, "track_request")
    assert {k: req[k] for k in ("op", "id", "z")} == {"op": "delete", "id": 4, "z": 0}
    w.track_request = req
    assert not (w.labels == 4).any()
    page.evaluate(f"window.mockModel.set('track_undo_ids', {w.track_undo_ids})")
    page.keyboard.press("Control+z")
    req = _get(page, "track_request")
    assert req["op"] == "undo" and req["serial"] == w.track_undo_ids[-1]
    w.track_request = req
    np.testing.assert_array_equal(w.labels, lab)
    page.keyboard.down("Shift")
    page.mouse.click(*_client_xy(page, 10, 10))
    page.keyboard.up("Shift")
    req = _get(page, "track_request")
    assert req["op"] == "delete" and req["id"] == 4 and req["z"] is None


def test_background_is_not_merged_split_or_picked_as_active_id(page, serve):
    """Clicking background with merge / split is refused locally; the picker arms the background brush."""
    init = np.zeros((H, W), np.uint32)
    init[0:5, :] = 65535
    init[20:25, 20:25] = 3
    w = _widget(labels=init, background=True)
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    active = _get(page, "active_label")
    page.keyboard.press("m")
    page.mouse.click(*_client_xy(page, 10, 2))
    assert not _get(page, "track_request")
    assert "annotated background" in page.locator(".lbl-status").inner_text()
    page.keyboard.press("p")
    page.mouse.click(*_client_xy(page, 10, 2))
    assert _get(page, "tool") == "background" and _get(page, "active_label") == active
    page.locator("input.lbl-label").fill("70000")
    page.locator("input.lbl-label").dispatch_event("change")
    assert _get(page, "active_label") == 65534, "the background id cannot be typed in as a cell id"


def test_counts_readout_shows_python_counts(page, serve):
    """The counts row shows ``counts_text`` and follows its changes."""
    init = np.zeros((H, W), np.uint32)
    init[0:2, :] = 65535
    init[20:25, 20:25] = 3
    w = _widget(labels=init, background=True)
    _open(page, serve(w))
    assert page.locator(".lbl-counts").inner_text() == w.counts_text
    assert "1 cell, 80 bg px, 105 annotated px" in w.counts_text
    page.evaluate("window.mockModel.set('counts_text', 'x')")
    assert page.locator(".lbl-counts").inner_text() == "x"


def _lab_rgba(page, x, y):
    return page.evaluate(
        f"""() => Array.from(document.querySelector('canvas.lbl-lab').getContext('2d').getImageData({x}, {y}, 1, 1).data)"""
    )


def test_outlined_display_draws_coloured_boundary_over_black_border(page, serve):
    """Outlined mode: id-4 square (5:15) gets a coloured edge, black band outside it, faint interior; key ``o`` toggles."""
    w, _ = _widget3d(display="outlined", outline_px=1, outline_border_px=2)
    w.z = 0
    _open(page, serve(w))
    edge = _lab_rgba(page, 5, 10)
    assert edge[3] == 255 and max(edge[:3]) > 0, edge
    border = _lab_rgba(page, 4, 10)
    assert border[:3] == [0, 0, 0] and border[3] == 255, border
    assert _lab_rgba(page, 2, 10)[3] == 0, "nothing beyond the border band"
    inner = _lab_rgba(page, 10, 10)
    assert 0 < inner[3] < 60, "interior only faintly tinted"
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("o")
    assert _get(page, "display") == "filled"
    assert _lab_rgba(page, 10, 10)[3] > 60
    page.keyboard.press("o")
    assert _get(page, "display") == "outlined"


def test_stack_keys_and_z_buttons(page, serve):
    """PageDown/PageUp step ``stack`` (wrapping); the z ‹ › buttons step ``z``."""
    img = np.random.default_rng(2).random((3, H, W)).astype(np.float32)
    w = LabelAnnotator([img, img.copy()], axes="ZYX", stack_names=["a", "b"])
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("PageDown")
    assert _get(page, "stack") == 1
    page.keyboard.press("PageDown")
    assert _get(page, "stack") == 0
    z0 = _get(page, "z")
    page.locator(".lbl-zrow .lbl-step").nth(3).click()  # z ›
    assert _get(page, "z") == min(z0 + 1, 2)


# ------------------------------------------------- review fixes (2026-11-25)
# what Python changes on a stack switch (``_on_stack``) and on ``set_labels``
_STACK_SWITCH = ["stack", "shape_zyx", "track_undo_ids", "z", "image_to_js", "labels_to_js",
                 "labels_prev_to_js", "labels_next_to_js", "labels_rev", "undo_epoch", "active_label"]
_SET_LABELS = ["track_undo_ids", "labels_to_js", "labels_prev_to_js", "labels_next_to_js", "labels_rev",
               "undo_epoch", "active_label"]
_TRACK_OP = ["track_undo_ids", "labels_to_js", "labels_prev_to_js", "labels_next_to_js", "labels_rev"]


def _push(page, w, names):
    """Mirror the Python traits ``names`` into the page's mock model, one ``set`` (and change event) each, in order.

    Hosts deliver the traits of one Python update in no fixed order (marimo
    emits one change per key), so tests pass several orders. Names the widget
    does not have are skipped.
    """
    st = json.loads(_state_json(w))
    for n in names:
        if n in st:
            page.evaluate(
                "([k, v]) => window.mockModel.set(k, (v && v.__b64__ !== undefined) ? window.b64ToDataView(v.__b64__) : v)",
                [n, st[n]],
            )


def _apply_last(page, w):
    """Apply the page's last ``labels_from_js`` message to the Python widget (as the host would)."""
    w.labels_from_js = base64.b64decode(page.evaluate("window.lastLabelsB64()"))


@pytest.mark.parametrize("epoch_first", [False, True])
def test_undo_after_stack_switch_does_not_restore_the_other_stack(page, serve, epoch_first):
    """ctrl+z after PageDown must not write the previous stack's slice into the shown stack (R4-01).

    Strategy: two 2D stacks (z is always 0); stack 1 holds a 100-px mask with
    id 5. Paint a stroke on stack 0 and apply it in Python, switch Python to
    stack 1 and mirror its traits into the page (``undo_epoch`` last or
    first), then press ctrl+z.
    Expected outcome: nothing is sent (the only snapshot belongs to stack 0),
    the status says there is nothing to undo, and the page still shows the
    id-5 mask. Before the fix ctrl+z sent stack 0's empty slice, which
    Python wrote into stack 1 and erased the mask.
    """
    img = np.random.default_rng(3).random((H, W)).astype(np.float32)
    lab1 = np.zeros((H, W), np.uint32)
    lab1[10:20, 10:20] = 5
    w = LabelAnnotator([img, img.copy()], labels=[None, lab1], brush_radius=0)
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(5, 30), (35, 30)])
    _apply_last(page, w)
    sent = page.evaluate("window.lastLabelsB64()")
    w.stack = 1
    order = ["undo_epoch"] + [n for n in _STACK_SWITCH if n != "undo_epoch"] if epoch_first else _STACK_SWITCH
    _push(page, w, order)
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Control+z")
    assert page.evaluate("window.lastLabelsB64()") == sent, "undo sent another stack's slice"
    assert "nothing to undo" in page.locator(".lbl-status").inner_text()
    assert (w.stack_labels[1] == 5).sum() == 100
    _stroke(page, [(5, 2), (8, 2)])  # the page still holds stack 1's mask
    z, lab = _last_labels(page)
    assert z == 0 and (lab == 5).sum() == 100 and lab[2, 5:9].all()


def test_undo_right_after_pagedown_does_not_send_the_previous_stack(page, serve):
    """ctrl+z after PageDown, before Python has answered the switch, sends nothing (R4-01).

    Strategy: two 2D stacks; stack 1 holds a 100-px mask with id 5. Paint a
    stroke on stack 0 and apply it in Python, press PageDown (the page's
    ``stack`` becomes 1, Python's answer is still in flight, e.g. busy kernel)
    and press ctrl+z at once. Then Python switches.
    Expected outcome: ctrl+z sends no label message (Python, which applies
    messages after the switch, would write the stack-0 snapshot into stack 1)
    and says there is nothing to undo; stack 1 keeps its mask. Negative
    control: the same ctrl+z without PageDown still undoes the stroke.
    """
    img = np.random.default_rng(3).random((H, W)).astype(np.float32)
    lab1 = np.zeros((H, W), np.uint32)
    lab1[10:20, 10:20] = 5
    w = LabelAnnotator([img, img.copy()], labels=[None, lab1], brush_radius=0)
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(5, 30), (35, 30)])
    _apply_last(page, w)
    sent = page.evaluate("window.lastLabelsB64()")
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("PageDown")
    assert _get(page, "stack") == 1
    page.keyboard.press("Control+z")
    assert page.evaluate("window.lastLabelsB64()") == sent, "undo sent stack 0's slice after PageDown"
    assert "nothing to undo" in page.locator(".lbl-status").inner_text()
    w.stack = 1
    assert (w.stack_labels[1] == 5).sum() == 100

    w2 = LabelAnnotator([img, img.copy()], labels=[None, lab1], brush_radius=0)
    w2.auto_fill = False
    _open(page, serve(w2))
    _stroke(page, [(5, 30), (35, 30)])
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Control+z")
    z, lab = _last_labels(page)
    assert z == 0 and not lab.any(), "negative control: ctrl+z undoes the stroke"


def test_undo_after_set_labels_does_not_revert_it(page, serve):
    """ctrl+z after ``set_labels`` must not restore a slice snapshot taken before it (R4-01).

    Strategy: paint a stroke and apply it in Python, then ``set_labels`` fills
    the slice with id 3 and the page receives Python's update; press ctrl+z.
    Expected outcome: nothing is sent and the page keeps showing id 3 (the
    snapshot predates ``set_labels``, whose docstring promises only to clear
    the track undo history; reverting the replacement on one slice is wrong).
    """
    w = _widget()
    w.auto_fill = False
    _open(page, serve(w))
    _stroke(page, [(5, 20), (35, 20)])
    _apply_last(page, w)
    sent = page.evaluate("window.lastLabelsB64()")
    w.set_labels(np.full((H, W), 3, np.uint32))
    _push(page, w, _SET_LABELS)
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Control+z")
    assert page.evaluate("window.lastLabelsB64()") == sent
    assert (w.labels == 3).all()


@pytest.mark.parametrize("ids_first", [False, True])
def test_track_undo_routes_to_python_on_every_stack(page, serve, ids_first):
    """Track ops on a second stack, and on the first after a round trip, stay undoable from the page (R4-02).

    Strategy: two 3D stacks with id 4 on z=0 and id 6 on z=1. On stack 0
    merge 4 -> 6 (Python), mirror the result; switch to stack 1 (traits in
    two orders) and merge there; ctrl+z; apply the request; switch back to
    stack 0; ctrl+z again.
    Expected outcome: each ctrl+z sends ``{op: undo, serial}`` for the shown
    stack's own last op and Python restores that stack exactly. Before the
    fix stack 1's first op reused serial 1, which the page had already seen,
    so it got no undo marker; and the round trip dropped stack 0's markers.
    """
    _, lab = _widget3d()
    img = np.random.default_rng(1).random((3, H, W)).astype(np.float32)
    w = LabelAnnotator([img, img.copy()], axes="ZYX", labels=[lab, lab.copy()], brush_radius=0)
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    switch = (["track_undo_ids"] + [n for n in _STACK_SWITCH if n != "track_undo_ids"]) if ids_first else _STACK_SWITCH
    assert w.merge_ids(4, 6)
    _push(page, w, _TRACK_OP)
    w.stack = 1
    _push(page, w, switch)
    assert w.merge_ids(4, 6)
    _push(page, w, _TRACK_OP)
    page.keyboard.press("Control+z")
    req = _get(page, "track_request")
    assert req.get("op") == "undo" and req["serial"] == w.track_undo_ids[-1]
    w.track_request = req
    np.testing.assert_array_equal(w.labels, lab)
    _push(page, w, _TRACK_OP)
    w.stack = 0
    _push(page, w, switch)
    page.keyboard.press("Control+z")
    req = _get(page, "track_request")
    assert req.get("op") == "undo" and req["serial"] == w.track_undo_ids[-1]
    w.track_request = req
    np.testing.assert_array_equal(w.labels, lab)


def _client_xy_in(page, view, x, y):
    r = page.evaluate(
        f"(() => {{ const b = document.querySelectorAll('canvas.lbl-img')[{view}].getBoundingClientRect();"
        " return [b.left, b.top, b.width, b.height]; })()"
    )
    return r[0] + (x + 0.5) * r[2] / W, r[1] + (y + 0.5) * r[3] / H


def _stroke_in(page, view, pts):
    page.mouse.move(*_client_xy_in(page, view, *pts[0]))
    page.mouse.down()
    for p in pts[1:]:
        page.mouse.move(*_client_xy_in(page, view, *p), steps=4)
    page.mouse.up()


def test_second_view_adopts_strokes_of_the_first(page, serve):
    """Two views of one model must not overwrite each other's strokes (R4-03).

    Strategy: render a second view of the same mock model next to the first
    (JupyterLab "new view for output", the widget displayed twice); paint row
    2 in view 1, then row 30 in view 2.
    Expected outcome: view 2's message holds both strokes (it adopted view
    1's message instead of sending its stale copy, which erased row 2 in
    Python), and view 1 shows view 2's stroke too.
    """
    page.set_viewport_size({"width": 1400, "height": 900})
    w = _widget()
    w.auto_fill = False
    _open(page, serve(w))
    page.evaluate(
        """() => { const box = document.createElement('div'), d = document.createElement('div');
        box.style.cssText = 'position:absolute; left:680px; top:0; width:630px';
        box.appendChild(d); document.body.appendChild(box);
        window.widget.render({ model: window.mockModel, el: d }); }"""
    )
    page.wait_for_function(
        """() => { const c = document.querySelectorAll('canvas.lbl-img');
        return c.length === 2 && c[1].width === 40 && document.querySelectorAll('.lbl-stage')[1].style.transform.includes('scale('); }"""
    )
    _stroke_in(page, 0, [(2, 2), (10, 2)])
    _stroke_in(page, 1, [(2, 30), (10, 30)])
    _, lab = _last_labels(page)
    assert lab[2, 2:11].all() and lab[30, 2:11].all(), "view 2 sent a stale slice without view 1's stroke"
    _stroke_in(page, 0, [(20, 20), (20, 20)])
    _, lab = _last_labels(page)
    assert lab[2, 2:11].all() and lab[30, 2:11].all() and lab[20, 20]


def test_view_rendered_after_strokes_resyncs_before_painting(page, serve):
    """A view rendered after strokes (page reload, re-display) asks Python for the current slice (R4-03).

    Strategy: apply a stroke on row 2 in Python (painted in an earlier view;
    Python does not echo strokes, so ``labels_to_js`` is stale), open a new
    page from the widget state, hand the page's ``labels_resync`` request to
    Python and mirror Python's answer; then paint row 30.
    Expected outcome: the page requested a resync, Python re-sent the slice,
    and the new stroke's message keeps row 2 (before the fix it was sent
    without it and Python lost the earlier stroke).
    """
    w = _widget()
    w.auto_fill = False
    edit = np.zeros((H, W), np.uint16)
    edit[2, 2:11] = 1
    w.labels_from_js = encode_label_slice(edit, 0)
    _open(page, serve(w))
    req = _get(page, "labels_resync")
    assert req, "the view did not ask Python to resync"
    w.labels_resync = req
    _push(page, w, ["labels_to_js", "labels_prev_to_js", "labels_next_to_js", "labels_rev"])
    _stroke(page, [(2, 30), (10, 30)])
    _apply_last(page, w)
    assert w.labels[2, 2:11].all() and w.labels[30, 2:11].all()


def test_z_and_stack_keys_are_ignored_while_painting(page, serve):
    """ArrowUp/Down, Alt+arrows and PageUp/Down do nothing during a drag (R4-05).

    Strategy: 3D widget at z=1; press the mouse on row 5, drag, press each
    navigation key, drag further and release.
    Expected outcome: ``z`` stays 1 and the whole stroke is sent for z=1.
    Afterwards ArrowUp steps z again (negative control).
    """
    w, _ = _widget3d()
    w.auto_fill = False
    _open(page, serve(w))
    page.locator(".lbl-viewport").focus()
    page.mouse.move(*_client_xy(page, 2, 5))
    page.mouse.down()
    page.mouse.move(*_client_xy(page, 15, 5), steps=4)
    for key in ("ArrowUp", "ArrowDown", "Alt+ArrowRight", "PageDown"):
        page.keyboard.press(key)
        assert _get(page, "z") == 1, key
    page.mouse.move(*_client_xy(page, 30, 5), steps=4)
    page.mouse.up()
    z, lab = _last_labels(page)
    assert z == 1 and lab[5, 2:31].all()
    page.keyboard.press("ArrowUp")
    assert _get(page, "z") == 2


def test_slice_replaced_mid_stroke_cancels_the_stroke(page, serve):
    """A slice reload from Python during a drag cancels the stroke instead of splitting it (R4-05).

    Strategy: 3D widget at z=1; start a drag on row 5, then Python moves to
    z=2 (traits mirrored into the page), continue the drag and release; then
    paint a dot. Repeat with a same-slice reload (``labels_rev`` bump) mid-drag.
    Expected outcome: no message carries the cancelled stroke (before the fix
    its first part was lost and the rest was sent as an edit of z=2); the
    status says it was cancelled; the following dot is sent for z=2 alone.
    """
    w, lab = _widget3d()
    w.auto_fill = False
    _open(page, serve(w))
    page.mouse.move(*_client_xy(page, 2, 5))
    page.mouse.down()
    page.mouse.move(*_client_xy(page, 15, 5), steps=4)
    w.z = 2
    _push(page, w, ["z", "image_to_js", "labels_to_js", "labels_prev_to_js", "labels_next_to_js"])
    assert "cancelled" in page.locator(".lbl-status").inner_text()  # before hovering overwrites it
    page.mouse.move(*_client_xy(page, 30, 5), steps=4)
    page.mouse.up()
    assert page.evaluate("window.lastLabelsB64()") is None
    _stroke(page, [(20, 38), (20, 38)])
    z, sent = _last_labels(page)
    assert z == 2 and not sent[5].any() and sent[38, 20] and sent.sum() == sent[38, 20]

    page.mouse.move(*_client_xy(page, 2, 5))
    page.mouse.down()
    page.mouse.move(*_client_xy(page, 15, 5), steps=4)
    before = page.evaluate("window.lastLabelsB64()")
    page.evaluate("window.mockModel.set('labels_rev', 5)")
    assert "cancelled" in page.locator(".lbl-status").inner_text()
    page.mouse.move(*_client_xy(page, 30, 5), steps=4)
    page.mouse.up()
    assert page.evaluate("window.lastLabelsB64()") == before


def test_view_cleanup_removes_window_listener_and_resize_observer(page, serve):
    """Removing a view must release its window ``resize`` listener and ResizeObserver (R4-18).

    Strategy: spy on ``window.add/removeEventListener`` and
    ``ResizeObserver.prototype.disconnect``, render a second view, call the
    cleanup function ``render`` returned.
    Expected outcome: the ``resize`` handler added by the view is removed and
    its observer disconnected (before the fix both kept every discarded view,
    with its image, labels and undo snapshots, alive).
    """
    w = _widget()
    _open(page, serve(w))
    out = page.evaluate(
        """() => {
        const added = [], removed = []; let disconnected = 0;
        const oa = window.addEventListener, orm = window.removeEventListener, od = ResizeObserver.prototype.disconnect;
        window.addEventListener = function (t, f, o) { if (t === 'resize') added.push(f); return oa.call(this, t, f, o); };
        window.removeEventListener = function (t, f, o) { if (t === 'resize') removed.push(f); return orm.call(this, t, f, o); };
        ResizeObserver.prototype.disconnect = function () { disconnected++; return od.call(this); };
        const d = document.createElement('div'); document.body.appendChild(d);
        const cleanup = window.widget.render({ model: window.mockModel, el: d });
        cleanup();
        window.addEventListener = oa; window.removeEventListener = orm; ResizeObserver.prototype.disconnect = od;
        return [added.length, removed.length, added[0] === removed[0], disconnected];
    }"""
    )
    assert out == [1, 1, True, 1]


# ------------------------------------------------- stage 2: channel controls, keys, wheel, paint cost
def _push_traits(page, w, names):
    """Copy Python traits into the mock model (what the host would sync), one ``set`` per trait."""
    for name in names:
        page.evaluate(f"window.mockModel.set({json.dumps(name)}, {json.dumps(getattr(w, name))})")


def test_channel_rows_follow_python_changes_in_data_units(page, serve):
    """Channel controls refresh in place on Python-side changes and work in data units (R4-20, R4-07).

    Strategy: a 2-channel 12-bit image; read the min/max sliders; change
    ``vmin``/``vmax``/``gamma``/``channel_colors``/``channel_visible`` in
    Python and push them to the page; then drag the first max slider with
    the keyboard (focus + ArrowLeft).
    Expected outcome: the sliders span the data (0..4095, step 1) and start
    at the default range; after the push every control shows the new value
    without the rows being rebuilt (the same DOM nodes; before, only the
    image re-composited and the controls stayed stale); ArrowLeft lowers
    ``vmax`` by one data unit (keyboard operable).
    """
    img = np.zeros((2, H, W), np.uint16)
    img[0] = np.arange(W) * 100
    img[1, ::2] = 4095
    w = LabelAnnotator(img, axes="CYX")
    _open(page, serve(w))
    rows = page.locator(".lbl-chan")
    assert rows.count() == 2
    hi0 = page.locator("input.lbl-vmax").first
    assert hi0.get_attribute("max") == "3900" and hi0.get_attribute("step") == "1"
    assert float(hi0.input_value()) == w.vmax[0]
    page.evaluate("window.firstRow = document.querySelector('.lbl-chan')")
    w.vmin, w.vmax, w.gamma = [100.0, 0.0], [2000.0, 4095.0], [0.5, 1.0]
    w.channel_colors = ["#ff0000", w.channel_colors[1]]
    w.channel_visible = [True, False]
    _push_traits(page, w, ["vmin", "vmax", "gamma", "channel_colors", "channel_visible", "data_range"])
    assert page.evaluate("window.firstRow === document.querySelector('.lbl-chan')"), "rows are refreshed, not rebuilt"
    assert page.locator("input.lbl-vmin").first.input_value() == "100"
    assert hi0.input_value() == "2000"
    assert page.locator("input.lbl-gamma").first.input_value() == "50"
    assert page.locator(".lbl-chan input[type=color]").first.input_value() == "#ff0000"
    assert page.locator(".lbl-chan input[type=checkbox]").nth(1).is_checked() is False
    hi0.focus()
    page.keyboard.press("ArrowLeft")
    assert _get(page, "vmax")[0] == 1999


def test_controls_have_accessible_names(page, serve):
    """Every input and select of the widget has an accessible name (R3-13).

    Strategy: render a 2-channel 3D widget with two stacks and list each
    input/select without an ``aria-label`` and outside a ``<label>``.
    Expected outcome: none (the channel sliders and toolbar boxes used to be
    announced without names).
    """
    img = np.zeros((2, 3, H, W), np.float32)
    w = LabelAnnotator([img, img], axes="CZYX")
    _open(page, serve(w))
    unnamed = page.evaluate(
        """() => [...document.querySelectorAll('#c input, #c select')]
            .filter((e) => !e.getAttribute('aria-label') && !e.closest('label'))
            .map((e) => e.className || e.type)"""
    )
    assert unnamed == []


def test_handled_keys_do_not_reach_the_host(page, serve):
    """Keys the widget handles stop at the viewport; others reach the host (D17, R4-06).

    Strategy: a document-level keydown listener (where JupyterLab's command
    mode listens) records keys while the focused viewport receives the
    annotator's single-letter keys, digits, brackets, arrows, PageDown and
    ctrl+z; then keys the widget does not handle (``q``, ctrl+s).
    Expected outcome (+): none of the handled keys reaches the document (they
    used to bubble: b, x, m, 0, ... are JupyterLab cell commands) and the
    viewport carries ``data-lm-suppress-shortcuts`` (Lumino skips shortcut
    matching inside it); the tools still switch. (−) unhandled keys and
    modified letters still reach the host.
    """
    w, _ = _widget3d()
    _open(page, serve(w))
    assert page.get_attribute(".lbl-viewport", "data-lm-suppress-shortcuts") is not None
    page.evaluate("window.hostKeys = []; document.addEventListener('keydown', (e) => window.hostKeys.push(e.key))")
    page.locator(".lbl-viewport").focus()
    for key in ("b", "x", "m", "s", "f", "e", "p", "o", "n", "0", "[", "]", "ArrowUp", "ArrowDown",
                "Alt+ArrowRight", "PageDown", "Control+z", "o"):
        page.keyboard.press(key)
    assert [k for k in page.evaluate("window.hostKeys") if k not in ("Control", "Alt", "Shift", "Meta")] == []
    assert _get(page, "tool") == "picker"
    page.evaluate("window.hostKeys = []")
    page.keyboard.press("q")
    page.keyboard.press("Control+s")
    assert page.evaluate("window.hostKeys") == ["q", "Control", "s"]


def _zoom(page):
    return page.evaluate("document.querySelector('.lbl-stage').style.transform")


def test_plain_wheel_zooms_only_after_the_viewport_is_focused(page, serve):
    """A plain wheel over an unfocused widget scrolls the page; zoom needs a click/focus or ctrl (D17, R4-16).

    Strategy: make the page taller than the window, put the pointer over the
    viewport while focus is elsewhere and turn the wheel; then the same with
    ctrl held; then focus the viewport and turn the plain wheel.
    Expected outcome: (−) unfocused plain wheel leaves the zoom alone and the
    page scrolls (the wheel used to always zoom: a scroll trap); (+) ctrl
    (trackpad pinch) zooms without focus; after focusing, the plain wheel
    zooms and does not scroll the page.
    """
    w = _widget()
    _open(page, serve(w))
    page.evaluate("document.body.appendChild(Object.assign(document.createElement('div'), {style: 'height: 3000px'}))")
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(*_client_xy(page, 20, 20))
    z0 = _zoom(page)
    page.mouse.wheel(0, 200)
    page.wait_for_timeout(200)
    assert _zoom(page) == z0, "unfocused plain wheel must not zoom"
    assert page.evaluate("window.scrollY") > 0, "the page scrolls"
    page.evaluate("window.scrollTo(0, 0)")
    page.mouse.move(*_client_xy(page, 20, 20))
    page.keyboard.down("Control")
    page.mouse.wheel(0, -200)
    page.keyboard.up("Control")
    z1 = _zoom(page)
    assert z1 != z0, "ctrl+wheel (pinch) zooms without focus"
    page.locator(".lbl-viewport").focus()
    y = page.evaluate("window.scrollY")
    page.mouse.move(*_client_xy(page, 20, 20))
    page.mouse.wheel(0, -200)
    page.wait_for_timeout(200)
    assert _zoom(page) != z1 and page.evaluate("window.scrollY") == y


_BIG = 2048


def _paint_latency(page, display):
    """Median ms per pointermove over 5 strokes of 30 moves on the big slice (handler time only)."""
    page.evaluate(f"window.mockModel.set('display', {json.dumps(display)})")
    page.evaluate(
        """() => {
        window.moveMs = [];
        let t0 = 0;
        window.addEventListener('pointermove', () => { t0 = performance.now(); }, true);
        window.addEventListener('pointermove', () => { window.moveMs.push(performance.now() - t0); });
    }"""
    )
    box = page.evaluate("(() => { const b = document.querySelector('canvas.lbl-img').getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })()")

    def at(x, y):
        return box[0] + (x + 0.5) * box[2] / _BIG, box[1] + (y + 0.5) * box[3] / _BIG

    runs = []
    for k in range(5):
        page.evaluate("window.moveMs = []")
        y = 300 + 300 * k
        page.mouse.move(*at(200, y))
        page.mouse.down()
        for i in range(1, 31):
            page.mouse.move(*at(200 + 20 * i, y + (i % 3)))
        page.mouse.up()
        ms = page.evaluate("window.moveMs")
        runs.append(float(np.median(ms)))
    return float(np.median(runs)), runs


def test_paint_latency_on_a_2048_slice_is_bounded(page, serve):
    """Painting redraws only the dirty rectangle, so a stroke on a 2048² slice stays interactive (R4-17).

    Strategy: a 2048 x 2048 two-channel slice with 400 masks, brush radius
    5; five strokes of 30 pointer moves each, timing the widget's
    pointermove handling in the page, in the filled and the outlined
    display. Then undo the last stroke and check the undo buffer size.
    Expected outcome: the median per-move time (median over the 5 runs,
    printed) is below a generous 10 ms in both displays (every move used to
    rebuild the whole 4-megapixel label image: 29 ms filled and 49 ms
    outlined per move here); the undo entry of a stroke holds only its bounding
    box (a few kB instead of a full 8 MB slice copy) and still restores it.
    """
    rng = np.random.default_rng(0)
    img = rng.random((2, _BIG, _BIG), dtype=np.float32)
    lab = np.zeros((_BIG, _BIG), np.uint32)
    for k in range(400):
        y, x = rng.integers(0, _BIG - 40, 2)
        lab[y:y + 30, x:x + 30] = k + 1
    w = LabelAnnotator(img, axes="CYX", labels=lab, brush_radius=5)
    w.auto_fill = False
    _open_big = serve(w)
    page.set_viewport_size({"width": 900, "height": 900})
    page.goto(_open_big)
    page.wait_for_function("window.ready === true")
    page.wait_for_function(f"document.querySelector('canvas.lbl-img').width === {_BIG}")
    page.wait_for_function("document.querySelector('.lbl-stage').style.transform.includes('scale(')")
    filled, runs_f = _paint_latency(page, "filled")
    outlined, runs_o = _paint_latency(page, "outlined")
    print(f"\npaint latency per move on a 2048^2 slice: filled {filled:.2f} ms {runs_f}, "
          f"outlined {outlined:.2f} ms {runs_o}")
    assert filled < 10 and outlined < 10
    nbytes = page.evaluate("document.getElementById('c').__lblDebug.undoBytes()")
    assert 0 < nbytes < 2 * _BIG * _BIG // 10, f"undo holds {nbytes} bytes for 10 strokes"
    before = _last_labels(page)[1]
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("Control+z")
    after = _last_labels(page)[1]
    changed = np.argwhere(before != after)
    assert changed.size
    assert changed[:, 0].min() >= 1490, "only the last stroke (y ~ 1500) was undone"


@pytest.mark.parametrize("display", ["filled", "outlined"])
def test_partial_redraw_matches_a_full_redraw(page, serve, display):
    """The dirty-rectangle redraw paints exactly what a full redraw would (R4-17).

    Strategy: masks next to each other, ``outline_px=3`` and a 2-px border
    (the widest stamp reach); paint and erase strokes that touch, split and
    cross them, then erase again and undo that; read the label canvas, then
    force a full redraw (re-set ``opacity``) and read it again.
    Expected outcome: identical RGBA everywhere (no stale outline or fill
    pixels left at the edge of the redrawn rectangles).
    """
    lab = np.zeros((H, W), np.uint32)
    lab[5:15, 5:15] = 1
    lab[5:15, 15:25] = 2
    lab[20:30, 8:30] = 3
    w = _widget(labels=lab)
    w.auto_fill = False
    w.display = display
    w.outline_px = 3
    w.outline_border_px = 2
    w.brush_radius = 1
    _open(page, serve(w))
    _stroke(page, [(2, 10), (35, 10)])
    page.locator(".lbl-viewport").focus()
    page.keyboard.press("e")
    _stroke(page, [(20, 2), (20, 35)])
    _stroke(page, [(2, 25), (35, 25)])
    page.keyboard.press("Control+z")
    read = """() => { const c = document.querySelector('canvas.lbl-lab');
        return Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data); }"""
    partial = np.asarray(page.evaluate(read))
    page.evaluate("window.mockModel.set('opacity', window.mockModel.get('opacity'))")
    full = np.asarray(page.evaluate(read))
    assert (full.reshape(H, W, 4)[..., 3] > 0).any()
    np.testing.assert_array_equal(partial, full)
