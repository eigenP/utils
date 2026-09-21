export default {
  render({ model, el }) {
    // Styles
    el.style.display = "flex";
    el.style.flexDirection = "row";
    el.style.gap = "10px";
    el.style.fontFamily = "sans-serif";

    // Create Image Container
    const imgContainer = document.createElement("div");
    imgContainer.style.flexShrink = "0"; 
    imgContainer.style.marginRight = "20px";

    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.display = "block";
    imgContainer.appendChild(img);

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

    // New Safe Coordinate Helpers
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

    function createRangeSlider(label, startTrait, endTrait, maxTrait, scaleTrait) {
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

      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      labelEl.style.fontSize = "12px";
      labelEl.style.fontWeight = "bold";

      const inputsRow = document.createElement("div");
      inputsRow.style.display = "flex";
      inputsRow.style.alignItems = "center";
      inputsRow.style.gap = "2px";

      const startInput = document.createElement("input");
      startInput.type = "number";
      startInput.style.width = "48px";
      startInput.style.fontSize = "11px";

      const sep = document.createElement("span");
      sep.textContent = "-";
      sep.style.fontSize = "11px";

      const endInput = document.createElement("input");
      endInput.type = "number";
      endInput.style.width = "48px";
      endInput.style.fontSize = "11px";

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
      trackBg.style.position = "absolute";
      trackBg.style.left = "0";
      trackBg.style.right = "0";
      trackBg.style.height = "6px";
      trackBg.style.backgroundColor = "#e0e0e0";
      trackBg.style.borderRadius = "3px";
      trackBg.style.pointerEvents = "none";

      const rangeBar = document.createElement("div");
      rangeBar.style.position = "absolute";
      rangeBar.style.height = "6px";
      rangeBar.style.backgroundColor = "#80b3ff";
      rangeBar.style.borderRadius = "3px";
      rangeBar.style.pointerEvents = "none";

      function createThumb(title, color, cursor, zIndex) {
        const thumb = document.createElement("div");
        thumb.style.position = "absolute";
        thumb.style.width = "14px";
        thumb.style.height = "14px";
        thumb.style.borderRadius = "50%";
        thumb.style.backgroundColor = color;
        thumb.style.border = "2px solid #ffffff";
        thumb.style.boxShadow = "0 1px 3px rgba(0,0,0,0.4)";
        thumb.style.cursor = cursor;
        thumb.style.zIndex = zIndex;
        thumb.style.transform = "translate(-50%, -50%)";
        thumb.style.top = "50%";
        thumb.title = title;
        return thumb;
      }

      const startThumb = createThumb("Start", "#d0d0d0", "ew-resize", "2");
      const middleThumb = createThumb("Translate Range", "#000000", "grab", "3");
      const endThumb = createThumb("End", "#d0d0d0", "ew-resize", "2");

      trackContainer.appendChild(trackBg);
      trackContainer.appendChild(rangeBar);
      trackContainer.appendChild(startThumb);
      trackContainer.appendChild(middleThumb);
      trackContainer.appendChild(endThumb);

      container.appendChild(topRow);
      container.appendChild(trackContainer);

      function getValFromX(clientX) {
        const rect = trackContainer.getBoundingClientRect();
        if (rect.width <= 0) return 0;
        const maxVal = model.get(maxTrait) || 1;
        const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        return Math.round(frac * maxVal);
      }

      function update() {
        const startVal = model.get(startTrait) || 0;
        const endVal = model.get(endTrait) || 0;
        const maxVal = model.get(maxTrait) || 1;
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;

        const startPct = Math.max(0, Math.min(100, (startVal / maxVal) * 100));
        const endPct = Math.max(0, Math.min(100, (endVal / maxVal) * 100));
        const midVal = (startVal + endVal) / 2;
        const midPct = Math.max(0, Math.min(100, (midVal / maxVal) * 100));

        startThumb.style.left = `${startPct}%`;
        endThumb.style.left = `${endPct}%`;
        middleThumb.style.left = `${midPct}%`;

        rangeBar.style.left = `${startPct}%`;
        rangeBar.style.width = `${endPct - startPct}%`;

        if (scale !== 1.0) {
          startInput.value = parseFloat((startVal * scale).toFixed(2));
          endInput.value = parseFloat((endVal * scale).toFixed(2));
          startInput.step = "0.01";
          endInput.step = "0.01";
        } else {
          startInput.value = startVal;
          endInput.value = endVal;
          startInput.step = "1";
        }
      }

      model.on(`change:${startTrait}`, update);
      model.on(`change:${endTrait}`, update);
      model.on(`change:${maxTrait}`, update);
      if (scaleTrait) {
        model.on(`change:${scaleTrait}`, update);
      }
      update();

      function setupDrag(thumb, onDrag, onStart, onEnd) {
        thumb.addEventListener("pointerdown", (e) => {
          e.preventDefault();
          e.stopPropagation();
          thumb.setPointerCapture(e.pointerId);
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

      setupDrag(startThumb, (e) => {
        const endVal = model.get(endTrait) || 0;
        let newStart = Math.max(0, Math.min(endVal, getValFromX(e.clientX)));
        if (newStart !== model.get(startTrait)) {
          model.set(startTrait, newStart);
          model.save_changes();
        }
      });

      setupDrag(endThumb, (e) => {
        const maxVal = model.get(maxTrait) || 1;
        const startVal = model.get(startTrait) || 0;
        let newEnd = Math.max(startVal, Math.min(maxVal, getValFromX(e.clientX)));
        if (newEnd !== model.get(endTrait)) {
          model.set(endTrait, newEnd);
          model.save_changes();
        }
      });

      setupDrag(
        middleThumb,
        (e) => {
          const maxVal = model.get(maxTrait) || 1;
          const startVal = model.get(startTrait) || 0;
          const endVal = model.get(endTrait) || 0;
          const span = endVal - startVal;
          const targetMid = getValFromX(e.clientX);
          let newStart = Math.round(targetMid - span / 2);
          let newEnd = newStart + span;

          if (newStart < 0) {
            newStart = 0;
            newEnd = Math.min(maxVal, span);
          } else if (newEnd > maxVal) {
            newEnd = maxVal;
            newStart = Math.max(0, maxVal - span);
          }

          if (newStart !== model.get(startTrait) || newEnd !== model.get(endTrait)) {
            model.set(startTrait, newStart);
            model.set(endTrait, newEnd);
            model.save_changes();
          }
        },
        () => { middleThumb.style.cursor = "grabbing"; },
        () => { middleThumb.style.cursor = "grab"; }
      );

      startInput.addEventListener("change", () => {
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;
        const endVal = model.get(endTrait) || 0;
        let val = Math.round(parseFloat(startInput.value) / scale);
        if (isNaN(val)) { update(); return; }
        val = Math.max(0, Math.min(endVal, val));
        model.set(startTrait, val);
        model.save_changes();
        update();
      });

      endInput.addEventListener("change", () => {
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;
        const maxVal = model.get(maxTrait) || 1;
        const startVal = model.get(startTrait) || 0;
        let val = Math.round(parseFloat(endInput.value) / scale);
        if (isNaN(val)) { update(); return; }
        val = Math.max(startVal, Math.min(maxVal, val));
        model.set(endTrait, val);
        model.save_changes();
        update();
      });

      return container;
    }

    const xRange = createRangeSlider("X Range", "x_start", "x_end", "x_max_pos", "sx");
    const yRange = createRangeSlider("Y Range", "y_start", "y_end", "y_max_pos", "sy");
    const zRange = createRangeSlider("Z Range", "z_start", "z_end", "z_max_pos", "sz");

    const saveContainer = document.createElement("div");
    saveContainer.style.display = "flex";
    saveContainer.style.gap = "10px";
    saveContainer.style.alignItems = "center";
    saveContainer.style.marginTop = "10px";

    const saveLabel = document.createElement("span");
    saveLabel.textContent = "Filename:";

    const saveInput = document.createElement("input");
    saveInput.type = "text";
    saveInput.value = model.get("save_filename");
    saveInput.addEventListener("change", () => {
      model.set("save_filename", saveInput.value);
      model.save_changes();
    });

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "💾 SVG";
    saveBtn.title = "Save plot as SVG";
    saveBtn.style.padding = "6px 12px";
    saveBtn.style.backgroundColor = "#e0e0e0";
    saveBtn.style.color = "#333";
    saveBtn.style.border = "1px solid #999";
    saveBtn.style.borderRadius = "4px";
    saveBtn.style.cursor = "pointer";
    saveBtn.style.fontWeight = "bold";
    saveBtn.addEventListener("mouseover", () => { saveBtn.style.backgroundColor = "#ccc"; });
    saveBtn.addEventListener("mouseout", () => { saveBtn.style.backgroundColor = "#e0e0e0"; });
    saveBtn.addEventListener("click", () => {
      let current = model.get("save_trigger");
      model.set("save_trigger", current + 1);
      model.save_changes();
    });

    const copyParamsBtn = document.createElement("button");
    copyParamsBtn.innerHTML = "📋";
    copyParamsBtn.title = "Copy parameters to clipboard";
    copyParamsBtn.style.padding = "6px 12px";
    copyParamsBtn.style.backgroundColor = "#e0e0e0";
    copyParamsBtn.style.color = "#333";
    copyParamsBtn.style.border = "1px solid #999";
    copyParamsBtn.style.borderRadius = "4px";
    copyParamsBtn.style.cursor = "pointer";
    copyParamsBtn.style.fontSize = "16px";
    copyParamsBtn.addEventListener("mouseover", () => { copyParamsBtn.style.backgroundColor = "#ccc"; });
    copyParamsBtn.addEventListener("mouseout", () => { copyParamsBtn.style.backgroundColor = "#e0e0e0"; });
    copyParamsBtn.addEventListener("click", () => {
      let current = model.get("copy_params_trigger") || 0;
      model.set("copy_params_trigger", current + 1);
      model.save_changes();
    });

    model.on("change:copy_params_string", () => {
      const str = model.get("copy_params_string");
      if (str && navigator.clipboard) {
        navigator.clipboard.writeText(str).then(() => {
          const originalHTML = copyParamsBtn.innerHTML;
          copyParamsBtn.innerHTML = "✅";
          setTimeout(() => { copyParamsBtn.innerHTML = originalHTML; }, 1500);
        }).catch(err => console.error("Failed to copy text: ", err));
      }
    });

    saveContainer.appendChild(saveLabel);
    saveContainer.appendChild(saveInput);
    saveContainer.appendChild(saveBtn);
    saveContainer.appendChild(copyParamsBtn);

    const hasAnnotation = model.get("annotation_mode") !== undefined;
    if (hasAnnotation) {
      const saveCsvLabel = document.createElement("span");
      saveCsvLabel.textContent = "CSV:";
      saveCsvLabel.style.marginLeft = "20px";

      const saveCsvInput = document.createElement("input");
      saveCsvInput.type = "text";
      saveCsvInput.value = model.get("save_csv_filename") || "points.csv";
      saveCsvInput.addEventListener("change", () => {
        model.set("save_csv_filename", saveCsvInput.value);
        model.save_changes();
      });

      const saveCsvBtn = document.createElement("button");
      saveCsvBtn.textContent = "Save Points as CSV";
      saveCsvBtn.style.padding = "6px 12px";
      saveCsvBtn.style.backgroundColor = "#e0e0e0";
      saveCsvBtn.style.color = "#333";
      saveCsvBtn.style.border = "1px solid #999";
      saveCsvBtn.style.borderRadius = "4px";
      saveCsvBtn.style.cursor = "pointer";
      saveCsvBtn.style.fontWeight = "bold";
      saveCsvBtn.addEventListener("mouseover", () => { saveCsvBtn.style.backgroundColor = "#ccc"; });
      saveCsvBtn.addEventListener("mouseout", () => { saveCsvBtn.style.backgroundColor = "#e0e0e0"; });
      saveCsvBtn.addEventListener("click", () => {
        let current = model.get("save_csv_trigger");
        model.set("save_csv_trigger", current + 1);
        model.save_changes();
      });

      saveContainer.appendChild(saveCsvLabel);
      saveContainer.appendChild(saveCsvInput);
      saveContainer.appendChild(saveCsvBtn);
    }

    el.appendChild(imgContainer);

    const controlsDiv = document.createElement("div");
    controlsDiv.style.flexGrow = "1";
    controlsDiv.style.minWidth = "300px";
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
    channelsContainer.style.fontSize = "12px";
    channelsContainer.style.overflowY = "auto";
    channelsContainer.style.maxHeight = "300px";

    function updateChannelsUI() {
      channelsContainer.innerHTML = "";
      const channelNames = model.get("channel_names");
      const channelDtypes = model.get("channel_dtypes");
      const channelColors = model.get("channel_colors");

      if (!channelNames || channelNames.length === 0) return;

      channelNames.forEach((name, index) => {
        const dtype = (channelDtypes && channelDtypes[index]) || "unknown";
        const color = (channelColors && channelColors[index]) || "black";

        const chDiv = document.createElement("div");
        chDiv.style.border = "1px solid #ccc";
        chDiv.style.padding = "5px";
        chDiv.style.borderRadius = "4px";
        chDiv.style.display = "flex";
        chDiv.style.flexDirection = "row";
        chDiv.style.alignItems = "center";
        chDiv.style.gap = "8px";

        const chHeader = document.createElement("strong");
        chHeader.textContent = `${index}:`;
        chHeader.style.color = color;
        chHeader.style.width = "15px";
        chDiv.appendChild(chHeader);

        const createNumberInput = (label, traitName, isFloat, minVal, maxVal, allowEmpty) => {
          const row = document.createElement("div");
          row.style.display = "flex";
          row.style.alignItems = "center";
          row.style.gap = "2px";

          const lbl = document.createElement("span");
          lbl.textContent = label;
          lbl.style.fontSize = "11px";

          const inp = document.createElement("input");
          inp.type = "text";
          inp.style.width = "35px";
          inp.style.fontSize = "11px";

          const updateInput = () => {
            const arr = model.get(traitName);
            if (arr && arr.length > index) {
              inp.value = arr[index];
            }
          };

          updateInput();
          model.on(`change:${traitName}`, updateInput);

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

        chDiv.appendChild(createNumberInput("vmin", "vmin_list", isFloatDtype, isFloatDtype ? undefined : 0, dtypeMax, true));
        chDiv.appendChild(createNumberInput("vmax", "vmax_list", isFloatDtype, isFloatDtype ? undefined : 0, dtypeMax, true));
        chDiv.appendChild(createNumberInput("gamma", "gamma_list", true, 0, 2.0, false));
        chDiv.appendChild(createNumberInput("opacity", "opacity_list", true, 0, 1, false));

        const histCanvas = document.createElement("canvas");
        histCanvas.width = 160;
        histCanvas.height = 30;
        histCanvas.style.marginLeft = "10px";
        histCanvas.style.border = "1px solid #ccc";
        histCanvas.style.borderRadius = "2px";
        histCanvas.style.backgroundColor = "#fff";
        chDiv.appendChild(histCanvas);

        const drawHistogram = () => {
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

          if (maxCount > 0) {
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.5;
            for (let i = 0; i < counts.length; i++) {
              const h = (counts[i] / maxCount) * histCanvas.height;
              const x = (i / counts.length) * histCanvas.width;
              const w = histCanvas.width / counts.length;
              ctx.fillRect(x, histCanvas.height - h, Math.ceil(w), h);
            }
            ctx.globalAlpha = 1.0;
          }

          const vmin_arr = model.get("vmin_list");
          const vmax_arr = model.get("vmax_list");
          const gamma_arr = model.get("gamma_list");

          let vmin = vmin_arr && vmin_arr[index] !== "" && vmin_arr[index] !== null ? parseFloat(vmin_arr[index]) : minData;
          let vmax = vmax_arr && vmax_arr[index] !== "" && vmax_arr[index] !== null ? parseFloat(vmax_arr[index]) : maxData;
          let gamma = gamma_arr && gamma_arr[index] !== undefined ? parseFloat(gamma_arr[index]) : 1.0;

          if (isNaN(vmin)) vmin = minData;
          if (isNaN(vmax)) vmax = maxData;
          if (vmax <= vmin) vmax = vmin + 1e-9;

          ctx.strokeStyle = "#000000";
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
        model.on("change:histograms_data", drawHistogram);
        model.on("change:vmin_list", drawHistogram);
        model.on("change:vmax_list", drawHistogram);
        model.on("change:gamma_list", drawHistogram);

        channelsContainer.appendChild(chDiv);
      });
    }

    updateChannelsUI();
    model.on("change:channel_names", updateChannelsUI);
    model.on("change:channel_colors", updateChannelsUI);

    const uiTogglesContainer = document.createElement("div");
    uiTogglesContainer.style.display = "flex";
    uiTogglesContainer.style.gap = "10px";
    uiTogglesContainer.style.alignItems = "center";
    uiTogglesContainer.style.marginBottom = "10px";

    const warningSpan = document.createElement("span");
    warningSpan.style.color = "red";
    warningSpan.style.fontSize = "14px";
    warningSpan.style.marginLeft = "auto";
    warningSpan.textContent = model.get("warning_msg");
    warningSpan.style.display = model.get("warning_msg") ? "block" : "none";

    model.on("change:warning_msg", () => {
      const msg = model.get("warning_msg");
      warningSpan.textContent = msg;
      warningSpan.style.display = msg ? "block" : "none";
    });

    const crosshairLabel = document.createElement("label");
    crosshairLabel.style.display = "flex";
    crosshairLabel.style.alignItems = "center";
    crosshairLabel.style.gap = "4px";
    crosshairLabel.style.fontSize = "14px";

    const crosshairCb = document.createElement("input");
    crosshairCb.type = "checkbox";
    crosshairCb.checked = model.get("show_crosshair");
    crosshairCb.addEventListener("change", () => {
      model.set("show_crosshair", crosshairCb.checked);
      model.save_changes();
    });

    model.on("change:show_crosshair", () => {
      crosshairCb.checked = model.get("show_crosshair");
    });

    crosshairLabel.appendChild(crosshairCb);
    crosshairLabel.appendChild(document.createTextNode("Show Crosshair"));
    uiTogglesContainer.appendChild(crosshairLabel);

    const syncLabel = document.createElement("label");
    syncLabel.style.display = "flex";
    syncLabel.style.alignItems = "center";
    syncLabel.style.gap = "4px";
    syncLabel.style.fontSize = "14px";
    syncLabel.style.marginLeft = "10px";

    const syncCb = document.createElement("input");
    syncCb.type = "checkbox";
    syncCb.checked = model.get("sync_on_hover");
    syncCb.addEventListener("change", () => {
      model.set("sync_on_hover", syncCb.checked);
      model.save_changes();
    });

    model.on("change:sync_on_hover", () => {
      syncCb.checked = model.get("sync_on_hover");
    });

    syncLabel.appendChild(syncCb);
    syncLabel.appendChild(document.createTextNode("Sync on Hover ('C')"));
    uiTogglesContainer.appendChild(syncLabel);

    // Annotation setup UI
    if (hasAnnotation) {
      const annotLabel = document.createElement("label");
      annotLabel.style.display = "flex";
      annotLabel.style.alignItems = "center";
      annotLabel.style.gap = "4px";
      annotLabel.style.fontSize = "14px";
      annotLabel.style.marginLeft = "20px";

      const annotCb = document.createElement("input");
      annotCb.type = "checkbox";
      annotCb.checked = model.get("annotation_mode");

      const actionSelect = document.createElement("select");
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
      annotLabel.appendChild(document.createTextNode("ANNOTATION"));

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

    const keydownListener = (e) => {
      if (!model.get("sync_on_hover")) return;
      if ((e.key === "c" || e.key === "C") && currentHoverCoords) {
        model.set("hover_coords", {
          ...currentHoverCoords,
          't': Date.now()
        });
        model.save_changes();
      }
    };

    document.addEventListener("keydown", keydownListener);

    model.on("destroy", () => {
      document.removeEventListener("keydown", keydownListener);
    });

    uiTogglesContainer.appendChild(warningSpan);

    controlsDiv.appendChild(uiTogglesContainer);
    controlsDiv.appendChild(channelsContainer);
    controlsDiv.appendChild(slidersContainer);
    controlsDiv.appendChild(saveContainer);

    el.appendChild(controlsDiv);

    function updateImage() {
      const src = model.get("image_data");
      if (src) {
        img.src = `data:image/png;base64,${src}`;
      }
    }

    model.on("change:image_data", updateImage);
    updateImage();
  }
};
