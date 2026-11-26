// Front end of the TNIA ortho viewers (TNIASliceWidget, TNIAAnnotatorWidget,
// TNIAScatterWidget). Conventions shared with LabelAnnotator (D14, D17):
// class-based dark theme (colours in tnia_plotting_anywidgets.css, geometry
// inline), focus- or hover-scoped keys that never fire while typing, a status
// line for feedback, labelled controls, keyboard-operable sliders.
export default {
  render({ model, el }) {
    // Layout: image and controls side by side, wrapping the controls below
    // the image when the row is too narrow; the image shrinks to fit (D14).
    el.classList.add("tnia-viewer");
    el.dataset.lmSuppressShortcuts = "true";  // typing in a box is not a notebook command
    el.style.display = "flex";
    el.style.flexDirection = "row";
    el.style.flexWrap = "wrap";
    el.style.alignItems = "flex-start";
    el.style.gap = "10px 20px";

    const imgContainer = document.createElement("div");
    imgContainer.style.flex = "0 1 auto";
    imgContainer.style.minWidth = "0";
    imgContainer.style.maxWidth = "100%";

    const img = document.createElement("img");
    img.alt = "orthogonal views (XY, ZY, XZ)";
    img.style.maxWidth = "100%";
    img.style.height = "auto";
    img.style.display = "block";
    imgContainer.appendChild(img);

    const fmt = (v) => parseFloat(Number(v).toPrecision(6));

    function button(text, title) {
      const b = document.createElement("button");
      b.className = "tnia-btn";
      b.type = "button";
      b.textContent = text;
      if (title) b.title = title;
      return b;
    }

    // Keys a slider thumb understands: +-1 step (arrows), +-big (Page keys), ends
    function sliderKeyDelta(e, big) {
      switch (e.key) {
        case "ArrowRight": case "ArrowUp": return 1;
        case "ArrowLeft": case "ArrowDown": return -1;
        case "PageUp": return big;
        case "PageDown": return -big;
        case "Home": return -Infinity;
        case "End": return Infinity;
        default: return 0;
      }
    }

    function makeThumb(title, ariaLabel, cursor, zIndex, size, colorVar) {
      const thumb = document.createElement("div");
      thumb.className = "tnia-thumb";
      thumb.style.position = "absolute";
      thumb.style.width = `${size}px`;
      thumb.style.height = `${size}px`;
      thumb.style.borderRadius = "50%";
      thumb.style.backgroundColor = colorVar;
      thumb.style.border = "2px solid #1e1e1e";
      thumb.style.boxShadow = "0 0 0 1px #888";
      thumb.style.cursor = cursor;
      thumb.style.zIndex = zIndex;
      thumb.style.transform = "translate(-50%, -50%)";
      thumb.style.top = "50%";
      thumb.title = title;
      // Keyboard and assistive technology (R3-13)
      thumb.tabIndex = 0;
      thumb.setAttribute("role", "slider");
      thumb.setAttribute("aria-label", ariaLabel);
      return thumb;
    }

    // Helper: Safely extract bounding box array [x0, y0, width, height]
    function getBbox(b) {
      if (!b) return null;
      if (Array.isArray(b)) return b;
      if (Array.isArray(b.bbox)) return b.bbox;
      if (b.x0 !== undefined && b.y0_js !== undefined && b.x1 !== undefined && b.y1_js !== undefined) {
        return [b.x0, b.y0_js, b.x1 - b.x0, b.y1_js - b.y0_js];
      }
      return null;
    }

    function getFractions(e, imgElement) {
      const rect = imgElement.getBoundingClientRect();
      const x_frac = (e.clientX - rect.left) / rect.width;
      const y_frac = (e.clientY - rect.top) / rect.height;
      return { x_frac, y_frac };
    }

    function findPlane(bounds, x_frac, y_frac) {
      if (!bounds) return null;
      const planes = ['xy', 'zy', 'xz'];
      for (const plane of planes) {
        const b = bounds[plane]; // bypasses Object.entries on proxies
        if (!b) continue;
        const bbox = getBbox(b);
        if (!bbox) continue;
        const [bx0, by0, bw, bh] = bbox;
        if (x_frac >= bx0 && x_frac <= bx0 + bw && y_frac >= by0 && y_frac <= by0 + bh) {
          return plane;
        }
      }
      return null;
    }

    // How the range boxes show an index i: (i + offset) * scale, in `unit`
    // (Python's range_display; R3-08). Without it: the old index x scale.
    function rangeDisplay(axis, scaleTrait) {
      const all = model.get("range_display");
      const rd = all && all[axis];
      if (rd) return rd;
      const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;
      return { offset: 0, scale, unit: "", calibrated: scale !== 1.0 };
    }

    function createRangeSlider(label, axis, startTrait, endTrait, maxTrait, scaleTrait) {
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.gap = "4px";
      container.style.minWidth = "180px";
      container.style.flex = "1";

      const topRow = document.createElement("div");
      topRow.style.display = "flex";
      topRow.style.justifyContent = "space-between";
      topRow.style.alignItems = "center";
      topRow.style.gap = "6px";

      const labelEl = document.createElement("label");
      labelEl.style.fontWeight = "bold";

      const inputsRow = document.createElement("div");
      inputsRow.style.display = "flex";
      inputsRow.style.alignItems = "center";
      inputsRow.style.gap = "2px";

      const startInput = document.createElement("input");
      startInput.type = "number";
      startInput.style.width = "64px";
      startInput.setAttribute("aria-label", `${label} start`);

      const sep = document.createElement("span");
      sep.textContent = "-";

      const endInput = document.createElement("input");
      endInput.type = "number";
      endInput.style.width = "64px";
      endInput.setAttribute("aria-label", `${label} end`);

      inputsRow.appendChild(startInput);
      inputsRow.appendChild(sep);
      inputsRow.appendChild(endInput);

      topRow.appendChild(labelEl);
      topRow.appendChild(inputsRow);

      const trackContainer = document.createElement("div");
      trackContainer.style.position = "relative";
      trackContainer.style.height = "24px";
      trackContainer.style.width = "100%";
      trackContainer.style.display = "flex";
      trackContainer.style.alignItems = "center";
      trackContainer.style.userSelect = "none";
      trackContainer.style.touchAction = "none";

      const trackBg = document.createElement("div");
      trackBg.className = "tnia-track";
      trackBg.style.position = "absolute";
      trackBg.style.left = "0";
      trackBg.style.right = "0";
      trackBg.style.height = "6px";
      trackBg.style.borderRadius = "3px";
      trackBg.style.pointerEvents = "none";

      const rangeBar = document.createElement("div");
      rangeBar.style.position = "absolute";
      rangeBar.style.height = "6px";
      rangeBar.style.backgroundColor = "#5b8def";
      rangeBar.style.borderRadius = "3px";
      rangeBar.style.pointerEvents = "none";

      const startThumb = makeThumb("Start", `${label} start`, "ew-resize", "2", 14, "#bbbbbb");
      const middleThumb = makeThumb("Translate Range", `${label} centre (moves the slab)`, "grab", "3", 14, "#2d6cdf");
      const endThumb = makeThumb("End", `${label} end`, "ew-resize", "2", 14, "#bbbbbb");

      trackContainer.appendChild(trackBg);
      trackContainer.appendChild(rangeBar);
      trackContainer.appendChild(startThumb);
      trackContainer.appendChild(middleThumb);
      trackContainer.appendChild(endThumb);

      container.appendChild(topRow);
      container.appendChild(trackContainer);

      // Largest index; 0 for a dimension of size 1 (Z of a 2D image), which
      // must not become 1 (`|| 1`) or the end could be set out of range.
      const getMax = () => model.get(maxTrait) ?? 0;

      function getValFromX(clientX) {
        const rect = trackContainer.getBoundingClientRect();
        if (rect.width <= 0) return 0;
        const maxVal = getMax();
        const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.round(frac * maxVal);
      }

      const show = (i, rd) => rd.calibrated ? fmt((i + rd.offset) * rd.scale) : i + rd.offset;
      const toIndex = (v, rd) => Math.round(rd.calibrated ? v / rd.scale - rd.offset : v - rd.offset);

      function update() {
        const startVal = model.get(startTrait) || 0;
        const endVal = model.get(endTrait) || 0;
        const maxVal = getMax();
        const rd = rangeDisplay(axis, scaleTrait);
        labelEl.textContent = rd.unit ? `${label} (${rd.unit})` : label;

        // Nothing to choose along a dimension of size 1: hide its row.
        container.style.display = maxVal > 0 ? "flex" : "none";
        const denom = Math.max(maxVal, 1);
        const startPct = Math.max(0, Math.min(100, (startVal / denom) * 100));
        const endPct = Math.max(0, Math.min(100, (endVal / denom) * 100));
        const midVal = (startVal + endVal) / 2;
        const midPct = Math.max(0, Math.min(100, (midVal / denom) * 100));

        startThumb.style.left = `${startPct}%`;
        endThumb.style.left = `${endPct}%`;
        middleThumb.style.left = `${midPct}%`;

        rangeBar.style.left = `${startPct}%`;
        rangeBar.style.width = `${endPct - startPct}%`;

        // Physical values with significant digits, one voxel per step;
        // indices with integer steps when uncalibrated (R3-08).
        startInput.value = show(startVal, rd);
        endInput.value = show(endVal, rd);
        const step = rd.calibrated ? String(fmt(rd.scale)) : "1";
        startInput.step = step;
        endInput.step = step;

        for (const [t, v] of [[startThumb, startVal], [middleThumb, midVal], [endThumb, endVal]]) {
          t.setAttribute("aria-valuemin", "0");
          t.setAttribute("aria-valuemax", String(maxVal));
          t.setAttribute("aria-valuenow", String(v));
          t.setAttribute("aria-valuetext", `${show(v, rd)}${rd.unit ? " " + rd.unit : ""}`);
        }
      }

      model.on(`change:${startTrait}`, update);
      model.on(`change:${endTrait}`, update);
      model.on(`change:${maxTrait}`, update);
      model.on("change:range_display", update);
      if (scaleTrait) {
        model.on(`change:${scaleTrait}`, update);
      }
      update();

      function setRange(newStart, newEnd) {
        if (newStart !== model.get(startTrait) || newEnd !== model.get(endTrait)) {
          model.set(startTrait, newStart);
          model.set(endTrait, newEnd);
          model.save_changes();
        }
      }
      function setStart(v) {
        const endVal = model.get(endTrait) || 0;
        setRange(Math.max(0, Math.min(endVal, v)), endVal);
      }
      function setEnd(v) {
        const startVal = model.get(startTrait) || 0;
        setRange(startVal, Math.max(startVal, Math.min(getMax(), v)));
      }
      // Move the slab so that its centre is near `targetMid`, keeping its width
      function translateTo(targetMid) {
        const maxVal = getMax();
        const startVal = model.get(startTrait) || 0;
        const endVal = model.get(endTrait) || 0;
        const span = endVal - startVal;
        let newStart = Math.round(targetMid - span / 2);
        let newEnd = newStart + span;
        if (newStart < 0) {
          newStart = 0;
          newEnd = Math.min(maxVal, span);
        } else if (newEnd > maxVal) {
          newEnd = maxVal;
          newStart = Math.max(0, maxVal - span);
        }
        setRange(newStart, newEnd);
      }
      function shiftBy(d) {
        const startVal = model.get(startTrait) || 0;
        const endVal = model.get(endTrait) || 0;
        const span = endVal - startVal;
        const maxVal = getMax();
        const newStart = Math.max(0, Math.min(maxVal - span, startVal + d));
        setRange(newStart, newStart + span);
      }

      function setupDrag(thumb, onDrag, onStart, onEnd) {
        thumb.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          thumb.setPointerCapture(e.pointerId);
          thumb.focus({ preventScroll: true });
          if (onStart) onStart();

          const handlePointerMove = (ev) => {
            onDrag(ev);
          };

          const handlePointerUp = (ev) => {
            thumb.releasePointerCapture(ev.pointerId);
            thumb.removeEventListener("pointermove", handlePointerMove);
            thumb.removeEventListener("pointerup", handlePointerUp);
            if (onEnd) onEnd();
          };

          thumb.addEventListener("pointermove", handlePointerMove);
          thumb.addEventListener("pointerup", handlePointerUp);
        });
      }

      setupDrag(startThumb, (e) => setStart(getValFromX(e.clientX)));
      setupDrag(endThumb, (e) => setEnd(getValFromX(e.clientX)));
      setupDrag(
        middleThumb,
        (e) => translateTo(getValFromX(e.clientX)),
        () => { middleThumb.style.cursor = "grabbing"; },
        () => { middleThumb.style.cursor = "grab"; }
      );

      // Keyboard (R3-13): the focused thumb takes arrows, Page keys, Home/End
      const bigStep = () => Math.max(1, Math.round(getMax() / 10));
      const keyHandler = (thumb, apply) => thumb.addEventListener("keydown", (e) => {
        const d = sliderKeyDelta(e, bigStep());
        if (!d || e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        e.stopPropagation();
        apply(d);
      });
      const clampD = (d) => (Number.isFinite(d) ? d : Math.sign(d) * (getMax() + 1));
      keyHandler(startThumb, (d) => setStart((model.get(startTrait) || 0) + clampD(d)));
      keyHandler(endThumb, (d) => setEnd((model.get(endTrait) || 0) + clampD(d)));
      keyHandler(middleThumb, (d) => shiftBy(clampD(d)));

      startInput.addEventListener("change", () => {
        const v = parseFloat(startInput.value);
        if (isNaN(v)) { update(); return; }
        setStart(toIndex(v, rangeDisplay(axis, scaleTrait)));
        update();
      });

      endInput.addEventListener("change", () => {
        const v = parseFloat(endInput.value);
        if (isNaN(v)) { update(); return; }
        setEnd(toIndex(v, rangeDisplay(axis, scaleTrait)));
        update();
      });

      return container;
    }

    const xRange = createRangeSlider("X Range", "x", "x_start", "x_end", "x_max_pos", "sx");
    const yRange = createRangeSlider("Y Range", "y", "y_start", "y_end", "y_max_pos", "sy");
    const zRange = createRangeSlider("Z Range", "z", "z_start", "z_end", "z_max_pos", "sz");

    // Status line (D15): Python's export outcome (status_msg) and local
    // copy feedback, announced politely to screen readers.
    const statusLine = document.createElement("div");
    statusLine.className = "tnia-status";
    statusLine.setAttribute("aria-live", "polite");
    function setStatus(msg, isError) {
      statusLine.textContent = msg || "";
      statusLine.classList.toggle("tnia-error", !!isError);
    }
    const showModelStatus = () => {
      const msg = model.get("status_msg") || "";
      setStatus(msg, msg.startsWith("error"));
    };
    model.on("change:status_msg", showModelStatus);
    showModelStatus();

    const saveContainer = document.createElement("div");
    saveContainer.style.display = "flex";
    saveContainer.style.flexWrap = "wrap";
    saveContainer.style.gap = "6px 10px";
    saveContainer.style.alignItems = "center";
    saveContainer.style.marginTop = "10px";

    const saveLabel = document.createElement("label");
    saveLabel.style.display = "flex";
    saveLabel.style.alignItems = "center";
    saveLabel.style.gap = "4px";
    saveLabel.appendChild(document.createTextNode("File:"));

    const saveInput = document.createElement("input");
    saveInput.type = "text";
    saveInput.style.width = "150px";
    saveInput.style.minWidth = "0";
    saveInput.value = model.get("save_filename");
    saveInput.title = "Format from the extension: .svg (editable text), .pdf or .png; an existing file is never overwritten";
    saveInput.addEventListener("change", () => {
      model.set("save_filename", saveInput.value);
      model.save_changes();
    });
    saveLabel.appendChild(saveInput);

    const saveBtn = button("💾 Save", "Save the figure to the file (format from its extension: svg, pdf, png; never overwrites, adds _1, _2, ...)");
    saveBtn.addEventListener("click", () => {
      setStatus("saving…");
      let current = model.get("save_trigger") || 0;
      model.set("save_trigger", current + 1);
      model.save_changes();
    });

    // Copy PNG: Python renders the figure (panels, channel labels, scale bar;
    // none of the GUI controls) at 300 dpi and sends it back as base64.
    // ClipboardItem is given a Promise synchronously inside the click, so the
    // write keeps the user activation while Python renders (Safari requires it).
    const COPY_PNG = "📋 PNG";
    const copyPngBtn = button(COPY_PNG, "Copy the figure to the clipboard as PNG (300 dpi)");

    let pendingPng = null;  // {resolve, reject, timer} while waiting on Python
    function flashCopyPng(text) {
      copyPngBtn.textContent = text;
      setTimeout(() => { copyPngBtn.textContent = COPY_PNG; }, 1500);
    }
    function base64ToPngBlob(b64) {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Blob([bytes], { type: "image/png" });
    }

    copyPngBtn.addEventListener("click", () => {
      if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
        console.error("Copy PNG: the Clipboard API is unavailable (needs a secure context).");
        setStatus("error: the clipboard is unavailable here (needs a secure context)", true);
        flashCopyPng("❌");
        return;
      }
      if (pendingPng) {
        // Drop its timer too: it would otherwise reset the newer pendingPng
        // to null when it fires (and outlive the view's cleanup).
        clearTimeout(pendingPng.timer);
        pendingPng.reject(new Error("superseded"));
      }
      const blobPromise = new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pendingPng = null;
          reject(new Error("timed out waiting for the PNG"));
        }, 30000);
        pendingPng = { resolve, reject, timer };
      });
      navigator.clipboard.write([new ClipboardItem({ "image/png": blobPromise })])
        .then(() => { flashCopyPng("✅"); setStatus("copied the figure (PNG) to the clipboard"); })
        .catch(err => {
          if (err && (err.message === "superseded" || err.message === "disposed")) return;
          console.error("Failed to copy PNG: ", err);
          setStatus(`error: PNG copy failed: ${err && err.message ? err.message : err}`, true);
          flashCopyPng("❌");
        });
      model.set("copy_png_trigger", (model.get("copy_png_trigger") || 0) + 1);
      model.save_changes();
    });

    model.on("change:copy_png_data", () => {
      const data = model.get("copy_png_data");
      if (!data || !pendingPng) return;
      const { resolve, reject, timer } = pendingPng;
      pendingPng = null;
      clearTimeout(timer);
      if (data === "error") reject(new Error("Python failed to render the PNG"));
      else resolve(base64ToPngBlob(data));
    });

    const COPY_KWARGS = "📋 kwargs";
    const copyParamsBtn = button(COPY_KWARGS, "Copy the current view settings to the clipboard as function kwargs");
    // Same pattern as the PNG button: the clipboard write starts inside the
    // click (keeping its user activation) with a Promise that resolves when
    // Python answers. Python clears the string before each answer, so every
    // click gets a change event even when the view is unchanged.
    let pendingKwargs = null;  // {resolve, reject, timer} while waiting on Python
    function flashCopyParams(text) {
      copyParamsBtn.textContent = text;
      setTimeout(() => { copyParamsBtn.textContent = COPY_KWARGS; }, 1500);
    }
    const kwargsCopied = () => { flashCopyParams("✅"); setStatus("copied the view kwargs to the clipboard"); };
    const kwargsFailed = (err) => {
      console.error("Failed to copy kwargs: ", err);
      setStatus(`error: kwargs copy failed: ${err && err.message ? err.message : err}`, true);
      flashCopyParams("❌");
    };
    copyParamsBtn.addEventListener("click", () => {
      if (navigator.clipboard && typeof ClipboardItem !== "undefined") {
        if (pendingKwargs) {
          clearTimeout(pendingKwargs.timer);  // see the PNG button
          pendingKwargs.reject(new Error("superseded"));
        }
        const blobPromise = new Promise((resolve, reject) => {
          const timer = setTimeout(() => {
            pendingKwargs = null;
            reject(new Error("timed out waiting for the kwargs"));
          }, 30000);
          pendingKwargs = { resolve, reject, timer };
        });
        navigator.clipboard.write([new ClipboardItem({ "text/plain": blobPromise })])
          .then(kwargsCopied)
          .catch(err => {
            if (err && (err.message === "superseded" || err.message === "disposed")) return;
            kwargsFailed(err);
          });
      }
      let current = model.get("copy_params_trigger") || 0;
      model.set("copy_params_trigger", current + 1);
      model.save_changes();
    });

    model.on("change:copy_params_string", () => {
      const str = model.get("copy_params_string");
      if (!str) return;
      if (pendingKwargs) {
        const { resolve, timer } = pendingKwargs;
        pendingKwargs = null;
        clearTimeout(timer);
        resolve(new Blob([str], { type: "text/plain" }));
        return;
      }
      // No click waiting (ClipboardItem unavailable, or the copy was triggered
      // from Python): plain text write.
      if (navigator.clipboard) {
        navigator.clipboard.writeText(str).then(kwargsCopied).catch(kwargsFailed);
      }
    });

    saveContainer.appendChild(saveLabel);
    saveContainer.appendChild(saveBtn);
    saveContainer.appendChild(copyPngBtn);
    saveContainer.appendChild(copyParamsBtn);

    const hasAnnotation = model.get("annotation_mode") !== undefined;
    if (hasAnnotation) {
      const saveCsvLabel = document.createElement("label");
      saveCsvLabel.style.display = "flex";
      saveCsvLabel.style.alignItems = "center";
      saveCsvLabel.style.gap = "4px";
      saveCsvLabel.appendChild(document.createTextNode("Points CSV:"));

      const saveCsvInput = document.createElement("input");
      saveCsvInput.type = "text";
      saveCsvInput.style.width = "120px";
      saveCsvInput.style.minWidth = "0";
      saveCsvInput.value = model.get("save_csv_filename") || "points.csv";
      saveCsvInput.addEventListener("change", () => {
        model.set("save_csv_filename", saveCsvInput.value);
        model.save_changes();
      });
      saveCsvLabel.appendChild(saveCsvInput);

      const saveCsvBtn = button("💾 Save points", "Save the points as CSV: voxel indices and physical positions (never overwrites)");
      saveCsvBtn.addEventListener("click", () => {
        setStatus("saving…");
        let current = model.get("save_csv_trigger") || 0;
        model.set("save_csv_trigger", current + 1);
        model.save_changes();
      });

      saveContainer.appendChild(saveCsvLabel);
      saveContainer.appendChild(saveCsvBtn);
    }

    el.appendChild(imgContainer);

    const controlsDiv = document.createElement("div");
    controlsDiv.style.flex = "1 1 320px";
    controlsDiv.style.minWidth = "min(300px, 100%)";
    controlsDiv.style.maxWidth = "100%";
    controlsDiv.style.display = "flex";
    controlsDiv.style.flexDirection = "column";
    controlsDiv.style.gap = "10px";

    const slidersContainer = document.createElement("div");
    slidersContainer.style.display = "flex";
    slidersContainer.style.flexDirection = "column";
    slidersContainer.style.gap = "10px";
    slidersContainer.style.width = "100%";
    slidersContainer.appendChild(xRange);
    slidersContainer.appendChild(yRange);
    slidersContainer.appendChild(zRange);

    const channelsContainer = document.createElement("div");
    channelsContainer.style.display = "flex";
    channelsContainer.style.flexDirection = "column";
    channelsContainer.style.gap = "10px";
    channelsContainer.style.overflowY = "auto";
    channelsContainer.style.maxHeight = "300px";

    // updateChannelsUI() discards the channel rows and rebuilds them from
    // scratch. The per-row refresh callbacks are collected here instead of
    // being re-subscribed to the model on every rebuild: the model offers no
    // way to unsubscribe, so re-subscribing would leave each previous
    // generation of listeners firing against detached DOM nodes forever.
    const inputRefreshers = [];   // { trait, fn } - one per channel input box
    const histogramRedraws = [];  // one per channel histogram canvas
    const histogramObservers = []; // ResizeObservers of the current rows

    function updateChannelsUI() {
      channelsContainer.innerHTML = "";
      inputRefreshers.length = 0;
      histogramRedraws.length = 0;
      for (const ro of histogramObservers) ro.disconnect();
      histogramObservers.length = 0;
      const channelNames = model.get("channel_names");
      const channelDtypes = model.get("channel_dtypes");
      const channelColors = model.get("channel_colors");
      const channelControls = model.get("channel_controls");

      if (!channelNames || channelNames.length === 0) return;

      channelNames.forEach((name, index) => {
        const dtype = (channelDtypes && channelDtypes[index]) || "unknown";
        const color = (channelColors && channelColors[index]) || "#bbbbbb";
        // 'opacity': vmin / vmax / gamma do nothing for this channel (R3-09)
        const onlyOpacity = !!(channelControls && channelControls[index] === "opacity");
        // The value a cleared vmin / vmax box stands for (R3-04)
        const autoV = (trait) => {
          const arr = model.get(trait === "vmin_list" ? "vmin_auto" : "vmax_auto");
          const v = arr ? arr[index] : undefined;
          return v === undefined || v === null ? undefined : Number(v);
        };

        const chDiv = document.createElement("div");
        chDiv.className = "tnia-chan";
        chDiv.style.padding = "5px";
        chDiv.style.display = "flex";
        chDiv.style.flexDirection = "row";
        chDiv.style.alignItems = "center";
        chDiv.style.gap = "8px";

        // Index and channel name (full name in the tooltip; R3-10)
        const chHeader = document.createElement("strong");
        chHeader.className = "tnia-cname";
        chHeader.textContent = `${index}: ${name}`;
        chHeader.title = name;
        chHeader.style.color = color;
        chHeader.style.maxWidth = "110px";
        chDiv.appendChild(chHeader);

        // `bound` = { trait, side }: keep this value at most ("max") or at least
        // ("min") the same channel's value of `trait`, as the histogram slider
        // keeps vmin <= vmax. An empty (auto) partner box does not bound it.
        const createNumberInput = (label, traitName, isFloat, minVal, maxVal, allowEmpty, bound) => {
          const row = document.createElement("div");
          row.style.display = "flex";
          row.style.alignItems = "center";
          row.style.gap = "2px";

          const lbl = document.createElement("span");
          lbl.className = "tnia-sub";
          lbl.textContent = label;
          lbl.style.fontSize = "11px";

          const inp = document.createElement("input");
          inp.type = "text";
          inp.style.width = "44px";
          inp.style.fontSize = "11px";
          inp.setAttribute("aria-label", `${label} of channel ${name}`);

          const updateInput = () => {
            const arr = model.get(traitName);
            if (arr && arr.length > index) {
              inp.value = arr[index];
            }
            if (allowEmpty) {
              const a = autoV(traitName);
              inp.placeholder = a === undefined ? "" : `auto ${fmt(a)}`;
              inp.title = a === undefined ? "" : `empty = automatic (${fmt(a)})`;
            }
          };

          updateInput();
          inputRefreshers.push({ trait: traitName, fn: updateInput });

          inp.addEventListener("change", () => {
            let val = inp.value.trim();
            if (val === "" && allowEmpty) {
              val = "";
            } else {
              val = isFloat ? parseFloat(val) : parseInt(val);
              if (isNaN(val)) {
                updateInput();
                return;
              }
              if (minVal !== undefined && val < minVal) val = minVal;
              if (maxVal !== undefined && val > maxVal) val = maxVal;
              if (bound) {
                const other = model.get(bound.trait);
                const raw = other ? other[index] : undefined;
                const o = raw === "" || raw === null || raw === undefined ? NaN : parseFloat(raw);
                if (!isNaN(o)) {
                  if (bound.side === "max" && val > o) val = o;
                  if (bound.side === "min" && val < o) val = o;
                }
              }
            }

            inp.value = val;
            const arr = [...model.get(traitName)];
            arr[index] = val;
            model.set(traitName, arr);
            model.save_changes();
          });

          row.appendChild(lbl);
          row.appendChild(inp);
          return row;
        };

        let dtypeMax = undefined;
        let isFloatDtype = false;
        if (dtype.includes("uint8")) dtypeMax = 255;
        else if (dtype.includes("uint16")) dtypeMax = 65535;
        else if (dtype.includes("float")) isFloatDtype = true;
        // Only unsigned and boolean data cannot go below 0 (int16 etc. can).
        const dtypeMin = (dtype.startsWith("uint") || dtype === "bool") ? 0 : undefined;

        if (!onlyOpacity) {
          chDiv.appendChild(createNumberInput("vmin", "vmin_list", isFloatDtype, dtypeMin, dtypeMax, true,
                                              { trait: "vmax_list", side: "max" }));
          chDiv.appendChild(createNumberInput("vmax", "vmax_list", isFloatDtype, dtypeMin, dtypeMax, true,
                                              { trait: "vmin_list", side: "min" }));
          // gamma 0 maps every pixel to full intensity; keep it strictly
          // positive. Python accepts any gamma > 0, so the box allows up to 10.
          chDiv.appendChild(createNumberInput("gamma", "gamma_list", true, 0.05, 10.0, false));
        }
        chDiv.appendChild(createNumberInput("opacity", "opacity_list", true, 0, 1, false));

        if (onlyOpacity) {
          channelsContainer.appendChild(chDiv);
          return;
        }

        // Histogram (fills the rest of the row, at least 160px) with a
        // vmin/vmax range slider underneath, styled like the X/Y/Z sliders.
        const histWrap = document.createElement("div");
        histWrap.style.display = "flex";
        histWrap.style.flexDirection = "column";
        histWrap.style.flex = "1 1 160px";
        histWrap.style.minWidth = "160px";

        const histCanvas = document.createElement("canvas");
        histCanvas.className = "tnia-hist";
        histCanvas.width = 160;
        histCanvas.height = 30;
        histCanvas.style.width = "100%";
        histCanvas.style.height = "30px";
        histCanvas.style.boxSizing = "border-box";
        histCanvas.style.borderRadius = "2px";
        histCanvas.setAttribute("role", "img");
        histCanvas.setAttribute("aria-label", `histogram of channel ${name} with the display curve`);
        histWrap.appendChild(histCanvas);

        const vTrack = document.createElement("div");
        vTrack.style.position = "relative";
        vTrack.style.height = "16px";
        vTrack.style.width = "100%";
        vTrack.style.userSelect = "none";
        vTrack.style.touchAction = "none";

        const vTrackBg = document.createElement("div");
        vTrackBg.className = "tnia-track";
        vTrackBg.style.position = "absolute";
        vTrackBg.style.left = "0";
        vTrackBg.style.right = "0";
        vTrackBg.style.top = "50%";
        vTrackBg.style.transform = "translateY(-50%)";
        vTrackBg.style.height = "4px";
        vTrackBg.style.borderRadius = "2px";
        vTrackBg.style.pointerEvents = "none";

        const vBar = document.createElement("div");
        vBar.style.position = "absolute";
        vBar.style.top = "50%";
        vBar.style.transform = "translateY(-50%)";
        vBar.style.height = "4px";
        vBar.style.backgroundColor = color;
        vBar.style.opacity = "0.6";
        vBar.style.borderRadius = "2px";
        vBar.style.pointerEvents = "none";

        const vminThumb = makeThumb("vmin", `vmin of channel ${name}`, "ew-resize", "2", 11, "#bbbbbb");
        const vmidThumb = makeThumb("Shift vmin and vmax", `display window of channel ${name}`, "grab", "3", 11, "#2d6cdf");
        const vmaxThumb = makeThumb("vmax", `vmax of channel ${name}`, "ew-resize", "2", 11, "#bbbbbb");
        vTrack.appendChild(vTrackBg);
        vTrack.appendChild(vBar);
        vTrack.appendChild(vminThumb);
        vTrack.appendChild(vmidThumb);
        vTrack.appendChild(vmaxThumb);
        histWrap.appendChild(vTrack);
        chDiv.appendChild(histWrap);

        // Histogram value range [lo, hi] for the slider. Integer channels have
        // bin edges half a level outside their levels.
        const isIntChannel = !isFloatDtype && dtype !== "unknown";
        const histRange = () => {
          const hists = model.get("histograms_data");
          const edges = hists && hists[index] && hists[index].bin_edges;
          if (!edges || edges.length < 2) return null;
          let lo = edges[0], hi = edges[edges.length - 1];
          if (isIntChannel) { lo = Math.ceil(lo); hi = Math.floor(hi); }
          return hi > lo ? [lo, hi] : null;
        };
        // While a thumb is dragged, the new vmin/vmax live here and only the
        // slider and tone curve follow; the model (and the Python render) is
        // updated once, on release.
        let pending = null;  // { vmin_list, vmax_list } raw values
        const rawV = (trait) => {
          if (pending) return pending[trait];
          const arr = model.get(trait);
          return arr ? arr[index] : undefined;
        };
        // A cleared box ("") is the automatic value Python renders with
        // (R3-04); the histogram edge only when Python sent none.
        const currentV = (trait, fallback) => {
          const raw = rawV(trait);
          const v = raw !== "" && raw !== null && raw !== undefined ? parseFloat(raw) : NaN;
          if (!isNaN(v)) return v;
          const a = autoV(trait);
          return a === undefined || isNaN(a) ? fallback : a;
        };
        const roundV = (v, lo, hi) => {
          if (isIntChannel) return Math.round(v);
          // Floats: snap to 1/1000 of the histogram range, without float noise.
          const step = (hi - lo) / 1000;
          return parseFloat((Math.round(v / step) * step).toPrecision(6));
        };
        const valFromX = (clientX, lo, hi) => {
          const rect = vTrack.getBoundingClientRect();
          const frac = rect.width > 0 ? Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) : 0;
          return lo + frac * (hi - lo);
        };
        const stageV = (vmin, vmax) => {
          pending = { vmin_list: vmin, vmax_list: vmax };
          drawHistogram();
        };
        const commitV = (vmin, vmax) => {
          const vminArr = [...model.get("vmin_list")];
          const vmaxArr = [...model.get("vmax_list")];
          if (vminArr[index] === vmin && vmaxArr[index] === vmax) return;
          vminArr[index] = vmin;
          vmaxArr[index] = vmax;
          model.set("vmin_list", vminArr);
          model.set("vmax_list", vmaxArr);
          model.save_changes();
        };
        const updateVSlider = () => {
          const r = histRange();
          vTrack.style.visibility = r ? "visible" : "hidden";
          if (!r) return;
          const [lo, hi] = r;
          const pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
          const vmin = currentV("vmin_list", lo);
          const vmax = currentV("vmax_list", hi);
          const a = pct(vmin);
          const b = pct(vmax);
          vminThumb.style.left = `${a}%`;
          vmaxThumb.style.left = `${b}%`;
          vmidThumb.style.left = `${(a + b) / 2}%`;
          vBar.style.left = `${Math.min(a, b)}%`;
          vBar.style.width = `${Math.abs(b - a)}%`;
          for (const [t, v] of [[vminThumb, vmin], [vmidThumb, (vmin + vmax) / 2], [vmaxThumb, vmax]]) {
            t.setAttribute("aria-valuemin", String(fmt(lo)));
            t.setAttribute("aria-valuemax", String(fmt(hi)));
            t.setAttribute("aria-valuenow", String(fmt(v)));
          }
        };
        const dragThumb = (thumb, onMove) => {
          thumb.addEventListener("pointerdown", (e) => {
            const r = histRange();
            if (!r) return;
            e.preventDefault();
            e.stopPropagation();
            thumb.setPointerCapture(e.pointerId);
            thumb.focus({ preventScroll: true });
            const start = { x: e.clientX, vmin: currentV("vmin_list", r[0]), vmax: currentV("vmax_list", r[1]) };
            const move = (ev) => onMove(ev, r[0], r[1], start);
            const up = (ev) => {
              thumb.releasePointerCapture(ev.pointerId);
              thumb.removeEventListener("pointermove", move);
              thumb.removeEventListener("pointerup", up);
              thumb.removeEventListener("pointercancel", up);
              if (thumb === vmidThumb) thumb.style.cursor = "grab";
              if (pending) {
                const p = pending;
                pending = null;
                commitV(p.vmin_list, p.vmax_list);
                drawHistogram();
              }
            };
            if (thumb === vmidThumb) thumb.style.cursor = "grabbing";
            thumb.addEventListener("pointermove", move);
            thumb.addEventListener("pointerup", up);
            thumb.addEventListener("pointercancel", up);
          });
        };
        dragThumb(vminThumb, (ev, lo, hi) => {
          const vmax = currentV("vmax_list", hi);
          stageV(roundV(Math.min(valFromX(ev.clientX, lo, hi), vmax), lo, hi), rawV("vmax_list"));
        });
        dragThumb(vmaxThumb, (ev, lo, hi) => {
          const vmin = currentV("vmin_list", lo);
          stageV(rawV("vmin_list"), roundV(Math.max(valFromX(ev.clientX, lo, hi), vmin), lo, hi));
        });
        dragThumb(vmidThumb, (ev, lo, hi, start) => {
          const rect = vTrack.getBoundingClientRect();
          if (rect.width <= 0) return;
          const span = start.vmax - start.vmin;
          let d = ((ev.clientX - start.x) / rect.width) * (hi - lo);
          // Keep the window inside the histogram range while it fits.
          d = Math.max(Math.min(d, hi - start.vmax), lo - start.vmin);
          if (span > hi - lo) d = 0;
          stageV(roundV(start.vmin + d, lo, hi), roundV(start.vmax + d, lo, hi));
        });

        // Keyboard (R3-13): one level (integers) or 1/100 of the range per
        // arrow, ten of them per Page key; committed at once.
        const vKeys = (thumb, apply) => thumb.addEventListener("keydown", (e) => {
          const r = histRange();
          if (!r) return;
          const d = sliderKeyDelta(e, 10);
          if (!d || e.ctrlKey || e.metaKey || e.altKey) return;
          e.preventDefault();
          e.stopPropagation();
          const [lo, hi] = r;
          const unit = isIntChannel ? 1 : (hi - lo) / 100;
          const delta = Number.isFinite(d) ? d * unit : Math.sign(d) * (hi - lo);
          apply(delta, lo, hi, currentV("vmin_list", lo), currentV("vmax_list", hi));
        });
        vKeys(vminThumb, (delta, lo, hi, vmin, vmax) =>
          commitV(roundV(Math.max(lo, Math.min(vmax, vmin + delta)), lo, hi), rawV("vmax_list")));
        vKeys(vmaxThumb, (delta, lo, hi, vmin, vmax) =>
          commitV(rawV("vmin_list"), roundV(Math.min(hi, Math.max(vmin, vmax + delta)), lo, hi)));
        vKeys(vmidThumb, (delta, lo, hi, vmin, vmax) => {
          const d = Math.max(Math.min(delta, hi - vmax), lo - vmin);
          if (vmax - vmin > hi - lo) return;
          commitV(roundV(vmin + d, lo, hi), roundV(vmax + d, lo, hi));
        });

        const drawHistogram = () => {
          updateVSlider();
          // Match the canvas resolution to its laid-out width.
          const cssW = Math.round(histCanvas.clientWidth);
          if (cssW > 0 && histCanvas.width !== cssW) histCanvas.width = cssW;

          const ctx = histCanvas.getContext("2d");
          ctx.clearRect(0, 0, histCanvas.width, histCanvas.height);

          const hists = model.get("histograms_data");
          if (!hists || !hists[index] || !hists[index].counts || hists[index].counts.length === 0) return;

          const counts = hists[index].counts;
          const edges = hists[index].bin_edges;
          const maxCount = Math.max(...counts);

          const minData = edges[0];
          const maxData = edges[edges.length - 1];
          const dataRange = maxData - minData;

          const bits = hists[index].bit_depth;
          histCanvas.title = bits
            ? `${bits}-bit histogram (${Math.round(minData + 0.5)} to ${Math.round(maxData - 0.5)})`
            : `Histogram (${minData.toPrecision(4)} to ${maxData.toPrecision(4)})`;

          if (maxCount > 0) {
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.6;
            for (let i = 0; i < counts.length; i++) {
              const h = (counts[i] / maxCount) * histCanvas.height;
              const x = (i / counts.length) * histCanvas.width;
              const w = histCanvas.width / counts.length;
              ctx.fillRect(x, histCanvas.height - h, Math.ceil(w), h);
            }
            ctx.globalAlpha = 1.0;
          }

          const gamma_arr = model.get("gamma_list");

          let vmin = currentV("vmin_list", minData);
          let vmax = currentV("vmax_list", maxData);
          let gamma = gamma_arr && gamma_arr[index] !== undefined ? parseFloat(gamma_arr[index]) : 1.0;

          if (isNaN(vmin)) vmin = minData;
          if (isNaN(vmax)) vmax = maxData;
          if (vmax <= vmin) vmax = vmin + 1e-9;

          ctx.strokeStyle = "#e6e6e6";
          ctx.lineWidth = 2;
          ctx.beginPath();

          for (let x = 0; x < histCanvas.width; x++) {
            const dataVal = minData + (x / histCanvas.width) * dataRange;
            let norm = (dataVal - vmin) / (vmax - vmin);
            if (norm < 0) norm = 0;
            if (norm > 1) norm = 1;

            let mapped = Math.pow(norm, gamma);
            const y = histCanvas.height - mapped * histCanvas.height;

            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        };

        drawHistogram();
        histogramRedraws.push(drawHistogram);
        if (typeof ResizeObserver !== "undefined") {
          const ro = new ResizeObserver(() => drawHistogram());
          ro.observe(histCanvas);
          histogramObservers.push(ro);
        }

        channelsContainer.appendChild(chDiv);
      });
    }

    updateChannelsUI();
    model.on("change:channel_names", updateChannelsUI);
    model.on("change:channel_colors", updateChannelsUI);
    model.on("change:channel_controls", updateChannelsUI);

    // Registered once; each fans out to whichever rows currently exist.
    const refreshInputs = (trait) => {
      for (const r of inputRefreshers) {
        if (r.trait === trait) r.fn();
      }
    };
    const redrawHistograms = () => {
      for (const draw of histogramRedraws) draw();
    };

    // vmin/vmax/gamma also move the tone curve drawn over the histogram.
    for (const trait of ["vmin_list", "vmax_list", "gamma_list"]) {
      model.on(`change:${trait}`, () => { refreshInputs(trait); redrawHistograms(); });
    }
    // The automatic values: box placeholders, slider and curve of cleared boxes
    model.on("change:vmin_auto", () => { refreshInputs("vmin_list"); redrawHistograms(); });
    model.on("change:vmax_auto", () => { refreshInputs("vmax_list"); redrawHistograms(); });
    model.on("change:opacity_list", () => refreshInputs("opacity_list"));
    model.on("change:histograms_data", redrawHistograms);

    const uiTogglesContainer = document.createElement("div");
    uiTogglesContainer.style.display = "flex";
    uiTogglesContainer.style.gap = "6px 14px";
    uiTogglesContainer.style.alignItems = "center";
    uiTogglesContainer.style.flexWrap = "wrap";

    const warningSpan = document.createElement("span");
    warningSpan.className = "tnia-warning";
    warningSpan.setAttribute("role", "status");
    warningSpan.style.marginLeft = "auto";
    warningSpan.textContent = model.get("warning_msg");
    warningSpan.style.display = model.get("warning_msg") ? "block" : "none";

    model.on("change:warning_msg", () => {
      const msg = model.get("warning_msg");
      warningSpan.textContent = msg;
      warningSpan.style.display = msg ? "block" : "none";
    });

    function checkbox(text, trait, title, isChecked) {
      const label = document.createElement("label");
      label.style.display = "flex";
      label.style.alignItems = "center";
      label.style.gap = "4px";
      if (title) label.title = title;
      const cb = document.createElement("input");
      cb.type = "checkbox";
      const refresh = () => { cb.checked = isChecked(model.get(trait)); };
      refresh();
      cb.addEventListener("change", () => {
        model.set(trait, cb.checked);
        model.save_changes();
      });
      model.on(`change:${trait}`, refresh);
      label.appendChild(cb);
      label.appendChild(document.createTextNode(text));
      return label;
    }

    uiTogglesContainer.appendChild(checkbox("Crosshair", "show_crosshair", "Draw the slab boundaries", (v) => !!v));
    uiTogglesContainer.appendChild(checkbox("Scale Bar", "scale_bar_show", "Draw the scale bar", (v) => v !== false));

    // Scale-bar font size in pt (empty = auto from the figure height, 0 hides the label, keeps the bar)
    const scaleBarSize = document.createElement("input");
    scaleBarSize.type = "text";
    scaleBarSize.title = "Scale bar label size (pt); empty = auto, 0 = no label (bar stays)";
    scaleBarSize.setAttribute("aria-label", "size of the scale-bar label in points");
    scaleBarSize.placeholder = "auto";
    scaleBarSize.style.width = "36px";
    scaleBarSize.style.fontSize = "11px";
    const refreshScaleBarSize = () => {
      const v = model.get("scale_bar_fontsize_pt");
      scaleBarSize.value = v === null || v === undefined ? "" : v;
    };
    refreshScaleBarSize();
    scaleBarSize.addEventListener("change", () => {
      const txt = scaleBarSize.value.trim();
      let val = null;
      if (txt !== "") {
        val = parseFloat(txt);
        if (isNaN(val)) { refreshScaleBarSize(); return; }
        val = Math.max(0, val);
      }
      model.set("scale_bar_fontsize_pt", val);
      model.save_changes();
      refreshScaleBarSize();
    });
    model.on("change:scale_bar_fontsize_pt", refreshScaleBarSize);
    const scaleBarSizeWrap = document.createElement("span");
    scaleBarSizeWrap.style.display = "flex";
    scaleBarSizeWrap.style.alignItems = "center";
    scaleBarSizeWrap.style.gap = "2px";
    scaleBarSizeWrap.style.fontSize = "11px";
    scaleBarSizeWrap.appendChild(scaleBarSize);
    scaleBarSizeWrap.appendChild(document.createTextNode("pt"));
    uiTogglesContainer.appendChild(scaleBarSizeWrap);

    // Hover sync is a key, not continuous hovering (R2-24)
    uiTogglesContainer.appendChild(checkbox(
      "Sync on 'C'", "sync_on_hover",
      "Point at a panel and press C (no Ctrl/Cmd/Alt, not while typing in a box) " +
      "to move the other slabs to the voxel under the pointer",
      (v) => !!v));

    // Annotation setup UI
    if (hasAnnotation) {
      const annotLabel = document.createElement("label");
      annotLabel.style.display = "flex";
      annotLabel.style.alignItems = "center";
      annotLabel.style.gap = "4px";
      annotLabel.title = "Click a panel to add or delete a point";

      const annotCb = document.createElement("input");
      annotCb.type = "checkbox";
      annotCb.checked = model.get("annotation_mode");

      const actionSelect = document.createElement("select");
      actionSelect.setAttribute("aria-label", "annotation action");
      actionSelect.disabled = !annotCb.checked;
      const addOpt = document.createElement("option");
      addOpt.value = "add";
      addOpt.textContent = "Add";
      const delOpt = document.createElement("option");
      delOpt.value = "delete";
      delOpt.textContent = "Delete";
      actionSelect.appendChild(addOpt);
      actionSelect.appendChild(delOpt);

      annotCb.addEventListener("change", () => {
        model.set("annotation_mode", annotCb.checked);
        model.save_changes();
        actionSelect.disabled = !annotCb.checked;
        img.style.cursor = annotCb.checked ? "crosshair" : "default";
      });

      model.on("change:annotation_mode", () => {
        annotCb.checked = model.get("annotation_mode");
        actionSelect.disabled = !annotCb.checked;
        img.style.cursor = annotCb.checked ? "crosshair" : "default";
      });

      annotLabel.appendChild(annotCb);
      annotLabel.appendChild(document.createTextNode("Annotate"));

      actionSelect.value = model.get("annotation_action");
      actionSelect.addEventListener("change", () => {
        model.set("annotation_action", actionSelect.value);
        model.save_changes();
      });

      model.on("change:annotation_action", () => {
        actionSelect.value = model.get("annotation_action");
      });

      uiTogglesContainer.appendChild(annotLabel);
      uiTogglesContainer.appendChild(actionSelect);

      if (model.get("annotation_mode")) {
        img.style.cursor = "crosshair";
      }
    }

    // ----------------------------------------------------
    // ROBUST MOUSE EVENT LISTENERS (Guaranteed attachment)
    // ----------------------------------------------------

    img.addEventListener("click", (e) => {
      if (!model.get("annotation_mode")) return; // Safe dynamic check

      const { x_frac, y_frac } = getFractions(e, img);
      const bounds = model.get("axis_bounds");
      const clicked_plane = findPlane(bounds, x_frac, y_frac);

      if (clicked_plane) {
        model.set("click_coords", {
          'plane': clicked_plane,
          'x': x_frac,
          'y': y_frac,
          't': Date.now()
        });
        model.save_changes();
      }
    });

    // Position under the pointer while it is over a panel; null otherwise,
    // so the 'C' key does nothing unless the pointer is on the image (D17).
    let currentHoverCoords = null;

    img.addEventListener("mousemove", (e) => {
      if (!model.get("sync_on_hover")) {
        currentHoverCoords = null;
        return;
      }

      const { x_frac, y_frac } = getFractions(e, img);
      const bounds = model.get("axis_bounds");
      const hover_plane = findPlane(bounds, x_frac, y_frac);

      if (hover_plane) {
        currentHoverCoords = {
          'plane': hover_plane,
          'x': x_frac,
          'y': y_frac
        };
      } else {
        currentHoverCoords = null;
      }
    });

    img.addEventListener("mouseleave", () => {
      currentHoverCoords = null;
    });

    // Typing in a text box (here or anywhere on the page) is never a command
    const isEditable = (t) => !!t && (t.isContentEditable ||
      /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || ""));

    // 'C' (D17 / R3-02): only while the pointer is over a panel, without
    // Ctrl/Cmd/Alt, and not while typing. A handled key is consumed so the
    // host (e.g. a notebook's command mode) does not also act on it; the
    // capture phase lets the widget see it first.
    const keydownListener = (e) => {
      if (e.key !== "c" && e.key !== "C") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (!currentHoverCoords || !model.get("sync_on_hover")) return;
      const target = (e.composedPath && e.composedPath()[0]) || e.target;
      if (isEditable(target)) return;
      e.preventDefault();
      e.stopPropagation();
      model.set("hover_coords", {
        ...currentHoverCoords,
        't': Date.now()
      });
      model.save_changes();
    };

    // Removed by the cleanup returned from render(): anywidget removes a
    // view's model listeners before calling it, so a 'destroy' handler would
    // never run, and the model outlives views of re-run cells.
    document.addEventListener("keydown", keydownListener, true);

    uiTogglesContainer.appendChild(warningSpan);

    controlsDiv.appendChild(uiTogglesContainer);
    controlsDiv.appendChild(channelsContainer);
    controlsDiv.appendChild(slidersContainer);
    controlsDiv.appendChild(saveContainer);
    controlsDiv.appendChild(statusLine);

    el.appendChild(controlsDiv);

    function updateImage() {
      const src = model.get("image_data");
      if (src) {
        img.src = `data:image/png;base64,${src}`;
      }
    }

    model.on("change:image_data", updateImage);
    updateImage();

    // View cleanup (anywidget calls it when the view is removed): release
    // everything registered outside this view's element, which would
    // otherwise keep the whole view (model, image, canvases) alive.
    return () => {
      document.removeEventListener("keydown", keydownListener, true);
      for (const ro of histogramObservers) ro.disconnect();
      histogramObservers.length = 0;
      for (const pending of [pendingPng, pendingKwargs]) {
        if (!pending) continue;
        clearTimeout(pending.timer);
        pending.reject(new Error("disposed"));
      }
      pendingPng = null;
      pendingKwargs = null;
    };
  }
};
