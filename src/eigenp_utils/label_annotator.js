// eigenp_utils LabelAnnotator front end (anywidget ESM).
// Wire format (little-endian):
//   image : uint32[4] (z, C, Y, X) + C*Y*X uint8
//   labels: uint32[3] (z, Y, X)    + Y*X uint16 (also labels_prev/next_to_js: slices z-1 / z+1)
// All compositing, painting and per-slice undo happen here; edited slice labels
// are sent back to Python after every stroke. Whole-track operations (merge /
// split an id across z, delete a whole mask) run in Python on the full volume (track_request); the
// undo stack holds markers for them (mirroring track_undo_ids) so undo routes
// to Python when the last operation was a track operation.

const UNDO_MAX = 30;
const MAX_LABEL = 65535; // labels travel as uint16: larger ids would wrap (65536 -> 0)
// reserved annotated-background id when the model's `background` is on (Python BACKGROUND_ID)
const BG_ID = 65535;
const BG_RGB_A = [112, 112, 112], BG_RGB_B = [48, 48, 48]; // hatched neutral greys, not in the cell palette
const KEY_HELP = "keys: b/e/f/p brush/eraser/fill/picker, g background brush (if enabled), " +
  "x delete mask (shift+click: all z), m/s merge/split track, n new id, [ ] radius, " +
  "ArrowUp/Down or Alt+ArrowRight/Left z+1/z-1, 0 fit, ctrl+z undo, space/right drag pan, wheel zoom, 1-9 class (semantic)";

function asBytes(v) {
  if (!v) return null;
  if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (v instanceof ArrayBuffer) return new Uint8Array(v);
  if (ArrayBuffer.isView(v)) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  return null;
}

function decodeImage(v) {
  const b = asBytes(v);
  if (!b || b.byteLength < 16) return null;
  const copy = b.slice(); // own, aligned buffer
  const h = new Uint32Array(copy.buffer, 0, 4);
  const [z, c, y, x] = h;
  return { z, c, y, x, data: new Uint8Array(copy.buffer, 16, c * y * x) };
}

function decodeLabels(v) {
  const b = asBytes(v);
  if (!b || b.byteLength < 12) return null;
  const copy = b.slice();
  const h = new Uint32Array(copy.buffer, 0, 3);
  const [z, y, x] = h;
  // header is 12 bytes, uint16 body needs 2-byte alignment: 12 is fine
  return { z, y, x, data: new Uint16Array(copy.buffer, 12, y * x) };
}

function encodeLabels(z, y, x, data) {
  const buf = new ArrayBuffer(12 + 2 * y * x);
  new Uint32Array(buf, 0, 3).set([z, y, x]);
  new Uint16Array(buf, 12, y * x).set(data);
  return new DataView(buf);
}

function hexToRgb(hex) {
  const h = (hex || "#ffffff").replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function instanceColor(id) {
  // golden-ratio hue walk, fixed S/V; deterministic per id
  const hue = ((id * 0.618033988749895) % 1 + 1) % 1;
  const s = 0.8, v = 1.0;
  const i = Math.floor(hue * 6), f = hue * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const rgb = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6];
  return rgb.map((u) => Math.round(u * 255));
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.appendChild(c);
  return e;
}

function render({ model, el: root }) {
  const state = {
    img: null, // {z,c,y,x,data}
    lab: null, // {z,y,x,data}
    prev: null, next: null, // labels of slices z-1 / z+1 (onion skin) or null
    undo: [],
    trackLive: new Set(), // track-op serials already mirrored into undo (Python's last track_undo_ids)
    epoch: model.get("undo_epoch") ?? 0, // Python's undo_epoch the slice snapshots belong to
    lastSent: null, // this view's last labels_from_js value (other views' messages are adopted)
    zoom: 1, panX: 0, panY: 0,
    painting: false, panning: false, spaceDown: false,
    last: null, stroke: null, strokeStart: null, strokeUndo: null,
    cursor: null, userView: false,
  };

  // ---------------------------------------------------------------- layout
  root.classList.add("lbl-annot");
  root.style.position = "relative";
  const toolbar = el("div", { class: "lbl-toolbar" });
  const chanPanel = el("div", { class: "lbl-channels" });
  const viewport = el("div", { class: "lbl-viewport", tabindex: "0" });
  const stage = el("div", { class: "lbl-stage" });
  const imgCanvas = el("canvas", { class: "lbl-img" });
  const labCanvas = el("canvas", { class: "lbl-lab" });
  const adjCanvas = el("canvas", { class: "lbl-adj" });
  const curCanvas = el("canvas", { class: "lbl-cur" });
  stage.append(imgCanvas, labCanvas, adjCanvas, curCanvas);
  viewport.appendChild(stage);
  // Geometry-critical styles inline, so painting coordinates stay correct even
  // if the host does not inject the stylesheet.
  Object.assign(viewport.style, { position: "relative", overflow: "hidden", touchAction: "none" });
  Object.assign(stage.style, { position: "absolute", left: "0", top: "0", transformOrigin: "0 0" });
  for (const c of [imgCanvas, labCanvas, adjCanvas, curCanvas]) {
    Object.assign(c.style, { position: "absolute", left: "0", top: "0", imageRendering: "pixelated" });
  }
  const zRow = el("div", { class: "lbl-zrow" });
  const statusBar = el("div", { class: "lbl-status" });
  const countsBar = el("div", { class: "lbl-counts", title: "annotation counts (current slice | whole volume)" });
  const main = el("div", { class: "lbl-main" }, [viewport, chanPanel]); // channel controls: column on the right
  root.append(toolbar, zRow, main, statusBar, countsBar);

  // toolbar controls
  const tools = ["brush", "eraser", "fill", "picker", "background", "delete", "merge", "split"];
  const toolTitles = {
    background: "background brush (g): paint annotated background (open scribbles, not filled)",
    delete: "delete mask (x): click a mask to remove it on this slice; shift+click: in ALL slices",
    merge: "merge track (m): click a mask; its id becomes the active id in ALL slices",
    split: "split track here (s): click a mask; its voxels on this and later slices get a new id",
  };
  const bgOn = () => !!model.get("background") && model.get("mode") !== "semantic";
  const maxCellId = () => (bgOn() ? BG_ID - 1 : MAX_LABEL);
  const toolBtns = {};
  for (const t of tools) {
    toolBtns[t] = el("button", { class: "lbl-btn", "data-tool": t, text: t, title: toolTitles[t] || `${t} (${t[0]})`, onclick: () => { model.set("tool", t); model.save_changes(); } });
    toolbar.appendChild(toolBtns[t]);
  }
  const labelInput = el("input", { type: "number", min: "0", max: String(MAX_LABEL), class: "lbl-label", title: "active label" });
  labelInput.addEventListener("change", () => {
    const v = Math.min(maxCellId(), Math.max(0, parseInt(labelInput.value || "0") || 0));
    labelInput.value = String(v);
    model.set("active_label", v); model.save_changes();
  });
  const classSelect = el("select", { class: "lbl-class" });
  classSelect.addEventListener("change", () => { model.set("active_label", parseInt(classSelect.value)); model.save_changes(); });
  const newBtn = el("button", { class: "lbl-btn", text: "new", title: "new label id (n)", onclick: () => { model.set("new_label_trigger", (model.get("new_label_trigger") || 0) + 1); model.save_changes(); } });
  const radius = el("input", { type: "range", min: "0", max: "50", class: "lbl-radius", title: "brush radius ([ / ])" });
  radius.addEventListener("input", () => { model.set("brush_radius", parseInt(radius.value)); model.save_changes(); });
  const opacity = el("input", { type: "range", min: "0", max: "100", class: "lbl-opacity", title: "label opacity" });
  opacity.addEventListener("input", () => { model.set("opacity", parseInt(opacity.value) / 100); model.save_changes(); });
  const mkToggle = (name, label, title) => {
    const cb = el("input", { type: "checkbox", class: `lbl-${name}` });
    cb.addEventListener("change", () => { model.set(name, cb.checked); model.save_changes(); });
    return [cb, el("label", { class: "lbl-toggle", title }, [cb, document.createTextNode(label)])];
  };
  const [fillCb, fillLbl] = mkToggle("auto_fill", "fill stroke", "close and fill each stroke");
  const [owCb, owLbl] = mkToggle("overwrite", "overwrite", "paint over other labels");
  // display toggle: filled <-> outlined (thick coloured boundary over a black border band); key o
  const olCb = el("input", { type: "checkbox", class: "lbl-display" });
  olCb.addEventListener("change", () => { model.set("display", olCb.checked ? "outlined" : "filled"); model.save_changes(); });
  const olLbl = el("label", { class: "lbl-toggle", title: "outlined labels: coloured boundary with a black border, no fill (o)" }, [olCb, document.createTextNode("outlined")]);
  const [adjCb, adjLbl] = mkToggle("show_adjacent", "z±1", "outlines of slice z-1 (dotted) and z+1 (faint), in each id's colour");
  // keep label = auto_increment off: the active id is not advanced after strokes / fills
  const keepCb = el("input", { type: "checkbox", class: "lbl-keep_label" });
  keepCb.addEventListener("change", () => { model.set("auto_increment", !keepCb.checked); model.save_changes(); });
  const keepLbl = el("label", { class: "lbl-toggle", title: "keep the active id after each stroke (paint one object across z)" }, [keepCb, document.createTextNode("keep label")]);
  const extentSpan = el("span", { class: "lbl-extent", title: "z extent of the active id" });
  const undoBtn = el("button", { class: "lbl-btn", text: "undo", title: "undo (ctrl+z)", onclick: () => undo() });
  const clearBtn = el("button", { class: "lbl-btn", text: "clear slice", onclick: () => clearSlice() });
  const fitBtn = el("button", { class: "lbl-btn", text: "fit", title: "reset zoom (0)", onclick: () => fit() });
  const chanBtn = el("button", { class: "lbl-btn lbl-active", text: "channels", title: "show / hide the channel controls", onclick: () => {
    const hide = chanPanel.style.display !== "none";
    chanPanel.style.display = hide ? "none" : "";
    chanBtn.classList.toggle("lbl-active", !hide);
    refit();
  } });
  toolbar.append(
    el("span", { class: "lbl-sep" }), labelInput, classSelect, newBtn, keepLbl, extentSpan,
    el("span", { class: "lbl-sep" }), el("span", { text: "r" }), radius, el("span", { text: "α" }), opacity,
    fillLbl, owLbl, olLbl, adjLbl, undoBtn, clearBtn, fitBtn, chanBtn,
    el("span", { class: "lbl-help", text: "?", title: KEY_HELP }),
  );

  const zSlider = el("input", { type: "range", min: "0", max: "0", class: "lbl-z" });
  const zLabel = el("span", { class: "lbl-zlabel" });
  zSlider.addEventListener("input", () => { model.set("z", parseInt(zSlider.value)); model.save_changes(); });
  const zPrev = el("button", { class: "lbl-btn lbl-step", text: "‹", title: "previous slice (ArrowDown)", onclick: () => stepZ(-1) });
  const zNext = el("button", { class: "lbl-btn lbl-step", text: "›", title: "next slice (ArrowUp)", onclick: () => stepZ(1) });
  const zGroup = el("span", { class: "lbl-group" }, [el("span", { text: "z" }), zPrev, zSlider, zNext, zLabel]);
  // stack selector (several 3D stacks, e.g. one per cage / field of view)
  const stackSel = el("select", { class: "lbl-stack", title: "stack (PageUp / PageDown)" });
  stackSel.addEventListener("change", () => { model.set("stack", parseInt(stackSel.value)); model.save_changes(); });
  const sPrev = el("button", { class: "lbl-btn lbl-step", text: "‹", title: "previous stack (PageUp)", onclick: () => stepStack(-1) });
  const sNext = el("button", { class: "lbl-btn lbl-step", text: "›", title: "next stack (PageDown)", onclick: () => stepStack(1) });
  const stackGroup = el("span", { class: "lbl-group" }, [el("span", { text: "stack" }), sPrev, stackSel, sNext]);
  zRow.append(stackGroup, zGroup);

  // ------------------------------------------------------------ channels UI
  function buildChannels() {
    chanPanel.innerHTML = "";
    const names = model.get("channel_names") || [];
    names.forEach((name, i) => {
      const vis = el("input", { type: "checkbox" });
      vis.checked = (model.get("channel_visible") || [])[i] !== false;
      vis.addEventListener("change", () => setListItem("channel_visible", i, vis.checked));
      const col = el("input", { type: "color", value: (model.get("channel_colors") || [])[i] || "#ffffff" });
      col.addEventListener("input", () => setListItem("channel_colors", i, col.value));
      const lo = el("input", { type: "range", min: "0", max: "255", value: String((model.get("vmin") || [])[i] ?? 0), title: "min" });
      const hi = el("input", { type: "range", min: "0", max: "255", value: String((model.get("vmax") || [])[i] ?? 255), title: "max" });
      const ga = el("input", { type: "range", min: "10", max: "300", value: String(Math.round(((model.get("gamma") || [])[i] ?? 1) * 100)), title: "gamma" });
      lo.addEventListener("input", () => setListItem("vmin", i, parseInt(lo.value)));
      hi.addEventListener("input", () => setListItem("vmax", i, parseInt(hi.value)));
      ga.addEventListener("input", () => setListItem("gamma", i, parseInt(ga.value) / 100));
      const slider = (txt, input) => el("label", { class: "lbl-sl" }, [el("span", { text: txt }), input]);
      chanPanel.appendChild(el("div", { class: "lbl-chan" }, [
        el("div", { class: "lbl-chan-head" }, [vis, col, el("span", { class: "lbl-cname", text: name, title: name })]),
        slider("min", lo), slider("max", hi), slider("γ", ga),
      ]));
    });
  }
  function setListItem(name, i, v) {
    const arr = [...(model.get(name) || [])];
    arr[i] = v;
    model.set(name, arr);
    model.save_changes();
    if (name !== "channel_colors" && name !== "channel_visible") return;
  }

  // -------------------------------------------------------------- drawing
  function compose() {
    const im = state.img;
    if (!im) return;
    if (imgCanvas.width !== im.x || imgCanvas.height !== im.y) {
      for (const c of [imgCanvas, labCanvas, adjCanvas, curCanvas]) { c.width = im.x; c.height = im.y; }
      fit();
    }
    const vis = model.get("channel_visible") || [];
    const cols = model.get("channel_colors") || [];
    const vmin = model.get("vmin") || [];
    const vmax = model.get("vmax") || [];
    const gam = model.get("gamma") || [];
    const n = im.x * im.y;
    const acc = new Float32Array(n * 3);
    for (let c = 0; c < im.c; c++) {
      if (vis[c] === false) continue;
      const [r, g, b] = hexToRgb(cols[c]);
      const lo = vmin[c] ?? 0, hi = Math.max((vmax[c] ?? 255), lo + 1), gm = gam[c] ?? 1;
      const lut = new Float32Array(256);
      for (let k = 0; k < 256; k++) lut[k] = Math.pow(Math.min(Math.max((k - lo) / (hi - lo), 0), 1), gm);
      const off = c * n;
      for (let p = 0; p < n; p++) {
        const v = lut[im.data[off + p]];
        if (v === 0) continue;
        acc[3 * p] += v * r; acc[3 * p + 1] += v * g; acc[3 * p + 2] += v * b;
      }
    }
    const ctx = imgCanvas.getContext("2d");
    const out = ctx.createImageData(im.x, im.y);
    for (let p = 0; p < n; p++) {
      out.data[4 * p] = Math.min(255, acc[3 * p]);
      out.data[4 * p + 1] = Math.min(255, acc[3 * p + 1]);
      out.data[4 * p + 2] = Math.min(255, acc[3 * p + 2]);
      out.data[4 * p + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }

  function labelRgb(id) {
    if (id === BG_ID && bgOn()) return BG_RGB_A;
    if (model.get("mode") === "semantic") {
      const cc = model.get("class_colors") || [];
      return hexToRgb(cc[id - 1] || "#ffffff");
    }
    const pal = model.get("instance_palette") || [];
    return pal.length ? hexToRgb(pal[(id - 1) % pal.length]) : instanceColor(id);
  }

  function isEdge(d, p, W, H, id) {
    const x = p % W, y = (p - x) / W;
    return x === 0 || y === 0 || x === W - 1 || y === H - 1 ||
      d[p - 1] !== id || d[p + 1] !== id || d[p - W] !== id || d[p + W] !== id;
  }

  function drawAdjacent() {
    // Onion skin: outlines of slice z-1 (dotted) and z+1 (faint), in each id's colour.
    const ctx = adjCanvas.getContext("2d");
    ctx.clearRect(0, 0, adjCanvas.width, adjCanvas.height);
    if (!model.get("show_adjacent") || !state.lab) return;
    const W = state.lab.x, H = state.lab.y;
    const out = ctx.createImageData(W, H);
    const cache = new Map();
    const bg = bgOn(); // background scribbles are not objects: no onion-skin outline
    for (const [lb, dotted] of [[state.next, false], [state.prev, true]]) {
      if (!lb || lb.x !== W || lb.y !== H) continue;
      const d = lb.data;
      for (let p = 0; p < d.length; p++) {
        const id = d[p];
        if (id === 0 || (id === BG_ID && bg) || !isEdge(d, p, W, H, id)) continue;
        if (dotted && ((p % W) + ((p / W) | 0)) % 2) continue;
        let rgb = cache.get(id);
        if (!rgb) { rgb = labelRgb(id); cache.set(id, rgb); }
        out.data[4 * p] = rgb[0]; out.data[4 * p + 1] = rgb[1]; out.data[4 * p + 2] = rgb[2];
        out.data[4 * p + 3] = dotted ? 255 : 110;
      }
    }
    ctx.putImageData(out, 0, 0);
  }

  function drawLabels() {
    const lb = state.lab;
    if (!lb) return;
    const ctx = labCanvas.getContext("2d");
    const out = ctx.createImageData(lb.x, lb.y);
    const a = Math.round(255 * (model.get("opacity") ?? 0.45));
    const outlined = model.get("display") === "outlined";
    const cache = new Map();
    const d = lb.data, W = lb.x, H = lb.y;
    const bg = bgOn();
    const bgA = Math.max(a, 140);
    const color = (id) => { let rgb = cache.get(id); if (!rgb) { rgb = labelRgb(id); cache.set(id, rgb); } return rgb; };
    for (let p = 0; p < d.length; p++) {
      const id = d[p];
      if (id === 0) continue;
      if (id === BG_ID && bg) {
        // annotated background: hatched neutral grey, in every display mode
        const x = p % W, y = (p - x) / W;
        const rgb = ((x + y) & 3) < 2 ? BG_RGB_A : BG_RGB_B;
        out.data[4 * p] = rgb[0]; out.data[4 * p + 1] = rgb[1]; out.data[4 * p + 2] = rgb[2];
        out.data[4 * p + 3] = bgA;
        continue;
      }
      if (outlined) {
        // interior: very faint fill (opacity slider / 4); the outline is added below
        const rgb = color(id);
        out.data[4 * p] = rgb[0]; out.data[4 * p + 1] = rgb[1]; out.data[4 * p + 2] = rgb[2];
        out.data[4 * p + 3] = a >> 2;
        continue;
      }
      if (model.get("outline_only") && !isEdge(d, p, W, H, id)) continue;
      const rgb = color(id);
      out.data[4 * p] = rgb[0]; out.data[4 * p + 1] = rgb[1]; out.data[4 * p + 2] = rgb[2];
      out.data[4 * p + 3] = model.get("outline_only") ? 255 : a;
    }
    if (outlined) {
      // Each non-background label's boundary pixels are stamped with a square of
      // half-width rc (coloured) over one of rc + rb (opaque black border band).
      // Cost: boundary pixels x (2r+1)^2, once per redraw, entirely in JS.
      const px = Math.max(1, Math.round(model.get("outline_px") ?? 2));
      const rc = (px - 1) >> 1, rc2 = px >> 1; // asymmetric for even widths
      const rb = Math.max(0, Math.round(model.get("outline_border_px") ?? 1));
      const oa = Math.min(255, Math.round(255 * Math.min(1, (a / 255) / 0.45)));
      const edge = [];
      for (let p = 0; p < d.length; p++) {
        const id = d[p];
        if (id === 0 || (id === BG_ID && bg) || !isEdge(d, p, W, H, id)) continue;
        edge.push(p);
      }
      const stamp = (r0, r1, fn) => {
        for (const p of edge) {
          const x = p % W, y = (p - x) / W;
          const ya = Math.max(0, y - r0), yb = Math.min(H - 1, y + r1);
          const xa = Math.max(0, x - r0), xb = Math.min(W - 1, x + r1);
          for (let yy = ya; yy <= yb; yy++) for (let xx = xa; xx <= xb; xx++) fn(yy * W + xx, p);
        }
      };
      stamp(rc + rb, rc2 + rb, (q) => { const o = 4 * q; out.data[o] = out.data[o + 1] = out.data[o + 2] = 0; out.data[o + 3] = oa; });
      stamp(rc, rc2, (q, p) => {
        const rgb = color(d[p]), o = 4 * q;
        out.data[o] = rgb[0]; out.data[o + 1] = rgb[1]; out.data[o + 2] = rgb[2]; out.data[o + 3] = oa;
      });
    }
    ctx.putImageData(out, 0, 0);
  }

  function drawCursor() {
    const ctx = curCanvas.getContext("2d");
    ctx.clearRect(0, 0, curCanvas.width, curCanvas.height);
    if (!state.cursor) return;
    const t = model.get("tool");
    if (t !== "brush" && t !== "eraser" && t !== "background") return;
    const r = model.get("brush_radius") ?? 2;
    ctx.strokeStyle = t === "eraser" ? "#ff4040" : t === "background" ? "#a0a0a0" : "#ffffff";
    ctx.lineWidth = Math.max(1 / state.zoom, 0.5);
    ctx.beginPath();
    ctx.arc(state.cursor[0] + 0.5, state.cursor[1] + 0.5, r + 0.5, 0, 2 * Math.PI);
    ctx.stroke();
  }

  function applyTransform() {
    stage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    drawCursor();
  }

  function fit() {
    if (!state.img) return;
    const w = viewport.clientWidth || 512;
    // available height: the window minus what sits above the viewport (toolbar,
    // channels, z row; offsetTop is scroll independent) and the status lines below
    const maxH = Math.max(320, Math.min(window.innerHeight * 0.96 - viewport.offsetTop - 44, 4000));
    state.zoom = Math.max(Math.min(w / state.img.x, maxH / state.img.y), 0.05);
    viewport.style.height = `${Math.round(state.img.y * state.zoom)}px`;
    state.panX = (w - state.img.x * state.zoom) / 2; state.panY = 0;
    state.userView = false;
    applyTransform();
  }
  const refit = () => { if (!state.userView) fit(); };
  window.addEventListener("resize", refit);
  const ro = typeof ResizeObserver === "function" ? new ResizeObserver(refit) : null;
  ro?.observe(root);
  ro?.observe(viewport);

  // ------------------------------------------------------------- painting
  function toPixel(ev) {
    const r = imgCanvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * imgCanvas.width);
    const y = Math.floor(((ev.clientY - r.top) / r.height) * imgCanvas.height);
    return [x, y];
  }

  function stamp(cx, cy) {
    const lb = state.lab;
    const r = model.get("brush_radius") ?? 2;
    const tool = model.get("tool");
    const eraser = tool === "eraser";
    const active = tool === "background" ? BG_ID : (model.get("active_label") ?? 1);
    const ow = !!model.get("overwrite");
    const r2 = (r + 0.5) * (r + 0.5);
    for (let dy = -r; dy <= r; dy++) {
      const y = cy + dy;
      if (y < 0 || y >= lb.y) continue;
      for (let dx = -r; dx <= r; dx++) {
        const x = cx + dx;
        if (x < 0 || x >= lb.x || dx * dx + dy * dy > r2) continue;
        const p = y * lb.x + x;
        state.stroke[p] = 1;
        if (eraser) {
          lb.data[p] = 0;
        } else if (ow || lb.data[p] === 0) {
          lb.data[p] = active;
        }
      }
    }
  }

  function line(x0, y0, x1, y1) {
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    for (;;) {
      stamp(x0, y0);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }

  function fillEnclosed() {
    // Fill pixels enclosed by this stroke: flood the stroke's padded bounding
    // box from its border through non-stroke pixels; unreached pixels are inside.
    const lb = state.lab, s = state.stroke, W = lb.x, H = lb.y;
    let x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (let p = 0; p < s.length; p++) if (s[p]) {
      const x = p % W, y = (p - x) / W;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (x1 < 0) return;
    x0 = Math.max(0, x0 - 1); y0 = Math.max(0, y0 - 1); x1 = Math.min(W - 1, x1 + 1); y1 = Math.min(H - 1, y1 + 1);
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    const seen = new Uint8Array(bw * bh);
    const stack = [];
    const push = (x, y) => {
      const q = (y - y0) * bw + (x - x0);
      if (seen[q] || s[y * W + x]) return;
      seen[q] = 1; stack.push(x, y);
    };
    for (let x = x0; x <= x1; x++) { push(x, y0); push(x, y1); }
    for (let y = y0; y <= y1; y++) { push(x0, y); push(x1, y); }
    while (stack.length) {
      const y = stack.pop(), x = stack.pop();
      if (x > x0) push(x - 1, y); if (x < x1) push(x + 1, y);
      if (y > y0) push(x, y - 1); if (y < y1) push(x, y + 1);
    }
    const active = model.get("active_label") ?? 1;
    const ow = !!model.get("overwrite");
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const q = (y - y0) * bw + (x - x0), p = y * W + x;
      if (!seen[q] && !s[p] && (ow || lb.data[p] === 0)) lb.data[p] = active;
    }
  }

  function floodFill(x, y) {
    const lb = state.lab, W = lb.x, H = lb.y;
    const target = lb.data[y * W + x], active = model.get("active_label") ?? 1;
    if (target === active) return false;
    const stack = [x, y];
    lb.data[y * W + x] = active;
    while (stack.length) {
      const cy = stack.pop(), cx = stack.pop();
      for (const [nx, ny] of [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]]) {
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const p = ny * W + nx;
        if (lb.data[p] !== target) continue;
        lb.data[p] = active;
        stack.push(nx, ny);
      }
    }
    return true;
  }

  function pushUndo() {
    const entry = { z: state.lab.z, data: state.lab.data.slice() };
    state.undo.push(entry);
    if (state.undo.length > UNDO_MAX) state.undo.shift();
    return entry;
  }

  function send() {
    const lb = state.lab;
    state.lastSent = encodeLabels(lb.z, lb.y, lb.x, lb.data);
    model.set("labels_from_js", state.lastSent);
    model.save_changes();
  }

  function onForeignLabels() {
    // Another view of this model (JupyterLab "new view for output", the widget
    // displayed twice, another browser tab) sent its edited slice. Adopt it, so
    // this view's next stroke does not send back a stale copy that erases it.
    // Pixels of a stroke in progress here keep this view's values.
    const v = model.get("labels_from_js");
    if (!v || v === state.lastSent || !state.lab) return;
    const lb = decodeLabels(v);
    if (!lb || lb.z !== state.lab.z || lb.x !== state.lab.x || lb.y !== state.lab.y) return;
    const d = state.lab.data, s = state.painting ? state.stroke : null;
    for (let p = 0; p < d.length; p++) if (!s || !s[p]) d[p] = lb.data[p];
    drawLabels();
  }

  function cancelStroke(why) {
    // The slice was replaced during a drag (Python changed z / stack / labels):
    // the stroke so far belongs to the old slice, so it is dropped rather than
    // split between two slices (or sent into another stack).
    if (!state.painting) return;
    state.painting = false;
    state.stroke = null;
    const i = state.undo.indexOf(state.strokeUndo);
    if (i >= 0) state.undo.splice(i, 1);
    state.strokeUndo = null;
    setStatus(`stroke cancelled: ${why}`);
  }

  function trackRequest(req) {
    // n continues from the model's last request so a re-rendered view cannot
    // repeat an identical dict (which the host would not deliver as a change)
    const n = Math.max(state.trackN || 0, (model.get("track_request") || {}).n || 0) + 1;
    state.trackN = n;
    model.set("track_request", { ...req, n });
    model.save_changes();
  }

  function syncTrackUndo() {
    // mirror Python's track-operation undo stack as markers in state.undo;
    // serials are unique per widget (never reused, across stacks too), so only
    // serials not mirrored yet are appended (in order), and a marker that fell
    // off the bounded undo stack is not re-added on top of later strokes
    // (which would undo that old op before them)
    const ids = model.get("track_undo_ids") || [];
    const live = new Set(ids);
    state.undo = state.undo.filter((u) => u.track === undefined || live.has(u.track));
    for (const s of ids) if (!state.trackLive.has(s)) state.undo.push({ track: s });
    state.trackLive = live;
    while (state.undo.length > UNDO_MAX) state.undo.shift();
  }

  function checkEpoch() {
    // Python bumps undo_epoch on a stack switch or set_labels: the slice
    // snapshots then belong to another stack or to replaced labels (ctrl+z
    // would write them into the shown stack). Drop them and rebuild the
    // track-op markers from the shown stack's track_undo_ids. Called from
    // every handler the update can reach first (hosts order keys freely).
    const ep = model.get("undo_epoch") ?? 0;
    if (ep === state.epoch) return false;
    state.epoch = ep;
    state.undo = [];
    state.trackLive = new Set();
    syncTrackUndo();
    return true;
  }

  function undo() {
    const lb = state.lab;
    if (!lb) return;
    for (let i = state.undo.length - 1; i >= 0; i--) {
      if (state.undo[i].track !== undefined) {
        // last operation was a whole-track merge/split: Python undoes it and
        // removes the marker via track_undo_ids
        trackRequest({ op: "undo", serial: state.undo[i].track });
        return;
      }
      if (state.undo[i].z === lb.z) {
        lb.data.set(state.undo[i].data);
        state.undo.splice(i, 1);
        drawLabels(); send();
        return;
      }
    }
    setStatus("nothing to undo on this slice");
  }

  function clearSlice() {
    if (!state.lab) return;
    pushUndo();
    state.lab.data.fill(0);
    drawLabels(); send();
  }

  function setStatus(msg) { statusBar.textContent = msg; }

  viewport.addEventListener("contextmenu", (e) => e.preventDefault());
  viewport.addEventListener("pointerdown", (ev) => {
    viewport.focus();
    if (!state.lab) return;
    if (ev.button === 1 || ev.button === 2 || state.spaceDown) {
      state.panning = true; state.last = [ev.clientX, ev.clientY];
      viewport.setPointerCapture?.(ev.pointerId);
      return;
    }
    if (ev.button !== 0) return;
    const [x, y] = toPixel(ev);
    if (x < 0 || y < 0 || x >= state.lab.x || y >= state.lab.y) return;
    const t = model.get("tool");
    if (t === "picker") {
      const id = state.lab.data[y * state.lab.x + x];
      // the background id is never an active (cell) id: pick the background brush instead
      if (id === BG_ID && bgOn()) model.set("tool", "background");
      else model.set("active_label", id);
      model.save_changes();
      return;
    }
    if (t === "delete") {
      const id = state.lab.data[y * state.lab.x + x];
      if (id === 0) { setStatus("delete: click on a mask"); return; }
      // shift+click: every slice; Python deletes (undoable via the track-op undo)
      trackRequest({ op: "delete", id, z: ev.shiftKey ? null : state.lab.z });
      return;
    }
    if (t === "merge" || t === "split") {
      const id = state.lab.data[y * state.lab.x + x];
      const active = model.get("active_label") ?? 0;
      if (id === 0) { setStatus(`${t}: click on a mask`); return; }
      if (id === BG_ID && bgOn()) { setStatus(`${t} refused: id ${BG_ID} is annotated background, not a cell`); return; }
      if (t === "split") { trackRequest({ op: "split", id, z: state.lab.z }); return; }
      if (active === 0) { setStatus("merge refused: active label is 0"); return; }
      if (active === id) { setStatus(`merge refused: clicked label ${id} is already the active label`); return; }
      trackRequest({ op: "merge", src: id, dst: active });
      return;
    }
    if (t === "fill") {
      pushUndo();
      if (floodFill(x, y)) { drawLabels(); send(); } else state.undo.pop();
      return;
    }
    state.strokeUndo = pushUndo();
    state.painting = true;
    state.stroke = new Uint8Array(state.lab.x * state.lab.y);
    state.last = [x, y];
    stamp(x, y);
    drawLabels();
    viewport.setPointerCapture?.(ev.pointerId);
  });
  viewport.addEventListener("pointermove", (ev) => {
    if (state.panning) {
      state.userView = true; state.panX += ev.clientX - state.last[0]; state.panY += ev.clientY - state.last[1];
      state.last = [ev.clientX, ev.clientY];
      applyTransform();
      return;
    }
    if (!state.lab) return;
    const [x, y] = toPixel(ev);
    state.cursor = [x, y];
    if (state.painting) {
      line(state.last[0], state.last[1], x, y);
      state.last = [x, y];
      drawLabels();
    }
    drawCursor();
    const inside = x >= 0 && y >= 0 && x < state.lab.x && y < state.lab.y;
    if (inside) setStatus(`z=${state.lab.z} y=${y} x=${x} label=${state.lab.data[y * state.lab.x + x]}`);
  });
  const endStroke = () => {
    if (state.panning) { state.panning = false; return; }
    if (!state.painting) return;
    state.painting = false;
    // only the cell brush closes and fills strokes; background strokes stay open scribbles
    if (model.get("tool") === "brush" && model.get("auto_fill")) fillEnclosed();
    state.stroke = null;
    drawLabels();
    send();
  };
  viewport.addEventListener("pointerup", endStroke);
  viewport.addEventListener("pointercancel", endStroke);
  viewport.addEventListener("pointerleave", () => { state.cursor = null; drawCursor(); });
  viewport.addEventListener("wheel", (ev) => {
    if (!state.img) return;
    ev.preventDefault();
    const r = viewport.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const f = Math.exp(-ev.deltaY * 0.0015);
    const nz = Math.min(Math.max(state.zoom * f, 0.05), 64);
    state.panX = mx - (mx - state.panX) * (nz / state.zoom);
    state.panY = my - (my - state.panY) * (nz / state.zoom);
    state.zoom = nz; state.userView = true;
    applyTransform();
  }, { passive: false });
  viewport.addEventListener("keydown", (ev) => {
    const k = ev.key;
    if (k === " ") { state.spaceDown = true; ev.preventDefault(); return; }
    if ((ev.ctrlKey || ev.metaKey) && k.toLowerCase() === "z") { undo(); ev.preventDefault(); return; }
    if (ev.altKey && (k === "ArrowLeft" || k === "ArrowRight")) {
      // Alt+Left/Right step z (and must not trigger browser back/forward);
      // z / stack keys do nothing during a stroke, which belongs to one slice
      ev.preventDefault(); ev.stopPropagation();
      if (!state.painting) stepZ(k === "ArrowRight" ? 1 : -1);
      return;
    }
    // letter keys only without modifiers: ctrl+s / ctrl+f / ctrl+p belong to the
    // host (saving the notebook must not silently arm the split tool)
    const mod = ev.ctrlKey || ev.metaKey || ev.altKey;
    if (k === "o" && !mod) { model.set("display", model.get("display") === "outlined" ? "filled" : "outlined"); model.save_changes(); return; }
    const map = { b: "brush", e: "eraser", f: "fill", p: "picker", x: "delete" };
    if (model.get("mode") !== "semantic") Object.assign(map, { m: "merge", s: "split" });
    if (bgOn()) map.g = "background";
    if (map[k]) { if (!mod) { model.set("tool", map[k]); model.save_changes(); } return; }
    if (k === "[" || k === "]") {
      const r = Math.max(0, (model.get("brush_radius") ?? 2) + (k === "]" ? 1 : -1));
      model.set("brush_radius", r); model.save_changes(); return;
    }
    if (k === "n" && !mod) { model.set("new_label_trigger", (model.get("new_label_trigger") || 0) + 1); model.save_changes(); return; }
    if (k === "0") { fit(); return; }
    if (/^[1-9]$/.test(k) && model.get("mode") === "semantic") {
      const id = parseInt(k);
      if (id <= (model.get("class_names") || []).length) { model.set("active_label", id); model.save_changes(); }
      return;
    }
    if (k === "PageUp" || k === "PageDown") { if (!state.painting) stepStack(k === "PageDown" ? 1 : -1); ev.preventDefault(); return; }
    if (k === "ArrowUp" || k === "ArrowDown") {
      if (!state.painting) stepZ(k === "ArrowUp" ? 1 : -1);
      ev.preventDefault();
    }
  });
  function stepStack(d) {
    const n = (model.get("stack_names") || []).length;
    if (n < 2) return;
    model.set("stack", (((model.get("stack") ?? 0) + d) % n + n) % n); model.save_changes();
  }
  function stepZ(dz) {
    const nz = (model.get("z") ?? 0) + dz;
    model.set("z", Math.min(Math.max(nz, 0), (model.get("shape_zyx") || [1])[0] - 1)); model.save_changes();
  }
  viewport.addEventListener("keyup", (ev) => { if (ev.key === " ") state.spaceDown = false; });

  // ------------------------------------------------------------ model sync
  function syncControls() {
    const t = model.get("tool");
    for (const [name, b] of Object.entries(toolBtns)) b.classList.toggle("lbl-active", name === t);
    const sem = model.get("mode") === "semantic";
    for (const e of [labelInput, newBtn, keepLbl, extentSpan, toolBtns.merge, toolBtns.split]) e.style.display = sem ? "none" : "";
    toolBtns.background.style.display = bgOn() ? "" : "none";
    labelInput.max = String(maxCellId());
    countsBar.textContent = model.get("counts_text") || "";
    classSelect.style.display = sem ? "" : "none";
    const names = model.get("class_names") || [];
    if (classSelect.options.length !== names.length + 1) {
      classSelect.innerHTML = "";
      classSelect.appendChild(el("option", { value: "0", text: "0: unlabelled" }));
      names.forEach((n, i) => classSelect.appendChild(el("option", { value: String(i + 1), text: `${i + 1}: ${n}` })));
    }
    const a = model.get("active_label") ?? 1;
    labelInput.value = String(a);
    classSelect.value = String(a);
    radius.value = String(model.get("brush_radius") ?? 2);
    opacity.value = String(Math.round(100 * (model.get("opacity") ?? 0.45)));
    fillCb.checked = !!model.get("auto_fill");
    owCb.checked = !!model.get("overwrite");
    olCb.checked = model.get("display") === "outlined";
    adjCb.checked = !!model.get("show_adjacent");
    keepCb.checked = !model.get("auto_increment");
    extentSpan.textContent = model.get("active_label_extent") || "";
    const shp = model.get("shape_zyx") || [1, 1, 1];
    zSlider.max = String(Math.max(0, shp[0] - 1));
    zSlider.value = String(model.get("z") ?? 0);
    zLabel.textContent = `${model.get("z") ?? 0} / ${Math.max(0, shp[0] - 1)}`;
    zGroup.style.display = shp[0] > 1 ? "" : "none";
    const sn = model.get("stack_names") || [];
    stackGroup.style.display = sn.length > 1 ? "" : "none";
    zRow.style.display = shp[0] > 1 || sn.length > 1 ? "" : "none";
    if (stackSel.options.length !== sn.length) {
      stackSel.innerHTML = "";
      sn.forEach((n, i) => stackSel.appendChild(el("option", { value: String(i), text: `${i + 1}/${sn.length} ${n}` })));
    }
    stackSel.value = String(model.get("stack") ?? 0);
    adjLbl.style.display = shp[0] > 1 ? "" : "none";
    const st = model.get("status");
    if (st) setStatus(st);
  }

  function onImage() {
    const im = decodeImage(model.get("image_to_js"));
    if (!im) return;
    if (state.lab && (im.z !== state.lab.z || im.x !== state.lab.x || im.y !== state.lab.y)) cancelStroke("the slice changed");
    state.img = im;
    compose();
  }
  function onLabels() {
    const lb = decodeLabels(model.get("labels_to_js"));
    if (!lb) return;
    checkEpoch();
    cancelStroke("the slice was reloaded from Python");
    if (!state.lab || state.lab.z !== lb.z) state.undo = state.undo.filter((u) => u.track !== undefined || u.z === lb.z);
    state.lab = lb;
    drawLabels();
    drawAdjacent();
  }
  function onAdjacent() {
    state.prev = decodeLabels(model.get("labels_prev_to_js"));
    state.next = decodeLabels(model.get("labels_next_to_js"));
    drawAdjacent();
  }

  const handlers = [
    ["change:image_to_js", onImage],
    ["change:labels_to_js change:labels_rev", onLabels],
    ["change:channel_visible change:channel_colors change:vmin change:vmax change:gamma", compose],
    ["change:labels_prev_to_js change:labels_next_to_js", onAdjacent],
    ["change:opacity change:outline_only change:display change:outline_px change:outline_border_px change:class_colors change:instance_palette change:mode change:background", () => { drawLabels(); drawAdjacent(); syncControls(); }],
    ["change:show_adjacent", () => { drawAdjacent(); syncControls(); }],
    ["change:track_undo_ids", () => { if (!checkEpoch()) syncTrackUndo(); }],
    ["change:undo_epoch", checkEpoch],
    // a stack switch requested here (PageDown, stack selector) reaches Python
    // later: drop the slice snapshots now, so a ctrl+z before Python's answer
    // cannot send the previous stack's slice into the new one (track markers
    // stay; Python refuses a serial that is not its top)
    ["change:stack", () => { state.undo = state.undo.filter((u) => u.track !== undefined); }],
    ["change:labels_from_js", onForeignLabels],
    ["change:tool change:active_label change:brush_radius change:auto_fill change:auto_increment change:overwrite change:z change:stack change:stack_names change:shape_zyx change:class_names change:status change:active_label_extent change:counts_text", () => { syncControls(); drawCursor(); }],
    ["change:channel_names", buildChannels],
  ];
  for (const [evt, fn] of handlers) for (const e of evt.split(" ")) model.on(e, fn);

  buildChannels();
  syncControls();
  onImage();
  onLabels();
  onAdjacent();
  syncTrackUndo();
  requestAnimationFrame(fit);
  // A view rendered after strokes (page reload, the widget displayed again, a
  // second view) starts from labels_to_js, which Python does not update per
  // stroke: ask Python to re-send the slice if it is stale. Without a kernel
  // (static export) there is no comm to sync with.
  try {
    model.set("labels_resync", (model.get("labels_resync") || 0) + 1);
    model.save_changes();
  } catch (e) { /* no live kernel */ }

  return () => {
    for (const [evt, fn] of handlers) for (const e of evt.split(" ")) model.off?.(e, fn);
    // a discarded view must not stay reachable (image, labels, undo snapshots)
    window.removeEventListener("resize", refit);
    ro?.disconnect();
    state.undo = [];
  };
}

export default { render };
