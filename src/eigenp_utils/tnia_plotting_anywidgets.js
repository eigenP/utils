export default {
  render({ model, el }) {
    // Styles
    el.style.display = "flex";
    el.style.flexDirection = "row";
    el.style.gap = "10px";
    el.style.fontFamily = "sans-serif";

    // Create Image Container

    const imgContainer = document.createElement("div");
    imgContainer.style.flexShrink = "0"; // Don't shrink below content size
    imgContainer.style.marginRight = "20px";

    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    img.style.display = "block";
    imgContainer.appendChild(img);


    function createSlider(label, traitName, minTrait, maxTrait, scaleTrait) {
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.flexDirection = "column"; // Stack label over slider+input for compactness
      container.style.alignItems = "flex-start";
      container.style.gap = "4px";
      container.style.minWidth = "80px";
      container.style.flex = "1";

      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      labelEl.style.fontSize = "12px";

      const controlRow = document.createElement("div");
      controlRow.style.display = "flex";
      controlRow.style.alignItems = "center";
      controlRow.style.gap = "4px";
      controlRow.style.width = "100%";

      const slider = document.createElement("input");
      slider.type = "range";
      slider.style.width = "100%";

      const numberInput = document.createElement("input");
      numberInput.type = "number";
      numberInput.style.width = "60px";
      numberInput.style.fontSize = "12px";

      function update() {
        const val = model.get(traitName);
        const min = model.get(minTrait) || 1;
        const max = model.get(maxTrait);
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;

        slider.min = min;
        slider.max = max;
        slider.value = val;

        const displayVal = val * scale;
        if (scale !== 1.0) {
            numberInput.value = parseFloat(displayVal.toFixed(2));
            numberInput.step = "0.01";
        } else {
            numberInput.value = val;
            numberInput.step = "1";
        }
      }

      update();

      model.on(`change:${traitName}`, update);
      model.on(`change:${minTrait}`, update);
      model.on(`change:${maxTrait}`, update);
      if (scaleTrait) {
        model.on(`change:${scaleTrait}`, update);
      }

      // Sync slider -> model
      slider.addEventListener("input", () => {
        model.set(traitName, parseInt(slider.value));
        model.save_changes();
      });

      // Sync number input -> model
      numberInput.addEventListener("change", () => {
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;
        const min = model.get(minTrait) || 1;
        const max = model.get(maxTrait);

        // Inverse scale to get integer index
        let newIndex = Math.round(parseFloat(numberInput.value) / scale);

        if (isNaN(newIndex)) {
            update();
            return;
        }

        // Clamp
        if (newIndex < min) newIndex = min;
        if (newIndex > max) newIndex = max;

        model.set(traitName, newIndex);
        model.save_changes();
        // Immediately re-update the UI to reflect clamped/rounded value
        update();
      });

      controlRow.appendChild(slider);
      controlRow.appendChild(numberInput);

      container.appendChild(labelEl);
      container.appendChild(controlRow);

      return container;
    }

    const xThick = createSlider("X Thickness", "x_t", "min_thickness", "x_thick_max", "sx");
    const yThick = createSlider("Y Thickness", "y_t", "min_thickness", "y_thick_max", "sy");
    const zThick = createSlider("Z Thickness", "z_t", "min_thickness", "z_thick_max", "sz");

    const xPos = createSlider("X Position", "x_s", "x_min_pos", "x_max_pos", "sx");
    const yPos = createSlider("Y Position", "y_s", "y_min_pos", "y_max_pos", "sy");
    const zPos = createSlider("Z Position", "z_s", "z_min_pos", "z_max_pos", "sz");

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
    saveBtn.textContent = "Save as SVG";
    saveBtn.style.padding = "6px 12px";
    saveBtn.style.backgroundColor = "#e0e0e0";
    saveBtn.style.color = "#333";
    saveBtn.style.border = "1px solid #999";
    saveBtn.style.borderRadius = "4px";
    saveBtn.style.cursor = "pointer";
    saveBtn.style.fontWeight = "bold";
    saveBtn.addEventListener("mouseover", () => {
        saveBtn.style.backgroundColor = "#ccc";
    });
    saveBtn.addEventListener("mouseout", () => {
        saveBtn.style.backgroundColor = "#e0e0e0";
    });
    saveBtn.addEventListener("click", () => {
      let current = model.get("save_trigger");
      model.set("save_trigger", current + 1);
      model.save_changes();
    });

    saveContainer.appendChild(saveLabel);
    saveContainer.appendChild(saveInput);
    saveContainer.appendChild(saveBtn);

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
        saveCsvBtn.addEventListener("mouseover", () => {
            saveCsvBtn.style.backgroundColor = "#ccc";
        });
        saveCsvBtn.addEventListener("mouseout", () => {
            saveCsvBtn.style.backgroundColor = "#e0e0e0";
        });
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

    // Thickness Sliders Container
    const thicknessContainer = document.createElement("div");
    thicknessContainer.style.display = "flex";
    thicknessContainer.style.gap = "15px";
    thicknessContainer.style.flexWrap = "wrap";
    thicknessContainer.style.alignItems = "center";
    thicknessContainer.appendChild(xThick);
    thicknessContainer.appendChild(yThick);
    thicknessContainer.appendChild(zThick);

    // Position Sliders Container
    const positionContainer = document.createElement("div");
    positionContainer.style.display = "flex";
    positionContainer.style.gap = "15px";
    positionContainer.style.flexWrap = "wrap";
    positionContainer.style.alignItems = "center";
    positionContainer.appendChild(xPos);
    positionContainer.appendChild(yPos);
    positionContainer.appendChild(zPos);



    // Channels Container
    const channelsContainer = document.createElement("div");
    channelsContainer.style.display = "flex";
    channelsContainer.style.flexDirection = "column";
    channelsContainer.style.gap = "10px";
    channelsContainer.style.fontSize = "12px";
    channelsContainer.style.overflowY = "auto";
    channelsContainer.style.maxHeight = "300px";

    const channelNames = model.get("channel_names");
    const channelDtypes = model.get("channel_dtypes");
    const channelColors = model.get("channel_colors");

    if (channelNames && channelNames.length > 0) {
      channelNames.forEach((name, index) => {
        const dtype = channelDtypes[index] || "unknown";
        const color = channelColors && channelColors.length > index ? channelColors[index] : "black";

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
          inp.type = "text"; // use text to easily handle empty string 'auto'
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
                // revert
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

        // Add Histogram Canvas
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

          // Draw curve
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






    const uiTogglesContainer = document.createElement("div");
    uiTogglesContainer.style.display = "flex";
    uiTogglesContainer.style.gap = "10px";
    uiTogglesContainer.style.alignItems = "center";
    uiTogglesContainer.style.marginBottom = "10px";

    const warningSpan = document.createElement("span");
    warningSpan.style.color = "red";
    warningSpan.style.fontSize = "14px";
    warningSpan.style.marginLeft = "auto"; // Push to right if in flex
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

    // Sync on hover toggle
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


    // Annotation Controls (only if annotator widget)
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
        annotCb.addEventListener("change", () => {
          model.set("annotation_mode", annotCb.checked);
          model.save_changes();
          actionSelect.disabled = !annotCb.checked;
          if(annotCb.checked) {
              img.style.cursor = "crosshair";
          } else {
              img.style.cursor = "default";
          }
        });

        model.on("change:annotation_mode", () => {
            annotCb.checked = model.get("annotation_mode");
            actionSelect.disabled = !annotCb.checked;
            img.style.cursor = annotCb.checked ? "crosshair" : "default";
        });

        annotLabel.appendChild(annotCb);
        annotLabel.appendChild(document.createTextNode("ANNOTATION"));

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

        // Add click listener for the image
        img.addEventListener("click", (e) => {
            if (!model.get("annotation_mode")) return;

            // e.offsetX and e.offsetY are relative to the padding edge of the target node
            const x_frac = e.offsetX / img.clientWidth;
            const y_frac = e.offsetY / img.clientHeight;

            // Map the click fraction to the axes.
            // We use the Python-computed axis_bounds to determine which axis was clicked.
            const bounds = model.get("axis_bounds");
            if (!bounds) return;

            let clicked_plane = null;
            // Bounds are [x0, y0, width, height] from 0 to 1 with origin at bottom-left in Matplotlib.
            // However, JS y_frac is from top-left.
            // Let's invert y_frac to match matplotlib's bottom-up coordinate system:
            const mpl_y_frac = 1.0 - y_frac;

            for (const [plane, b] of Object.entries(bounds)) {
                const [bx0, by0, bw, bh] = b;
                if (x_frac >= bx0 && x_frac <= bx0 + bw && mpl_y_frac >= by0 && mpl_y_frac <= by0 + bh) {
                    clicked_plane = plane;
                    break;
                }
            }

            if (clicked_plane) {
                // Send click directly to python
                // We add a timestamp so that consecutive identical clicks still trigger the observer
                model.set("click_coords", {
                    'plane': clicked_plane,
                    'x': x_frac,
                    'y': y_frac,
                    't': Date.now()
                });
                model.save_changes();
            }
        });

        // Initial cursor state
        if(model.get("annotation_mode")) {
            img.style.cursor = "crosshair";
        }
    }

    // Hover + 'C' key sync logic
    let currentHoverCoords = null;

    img.addEventListener("mousemove", (e) => {
        if (!model.get("sync_on_hover")) {
            currentHoverCoords = null;
            return;
        }

        const x_frac = e.offsetX / img.clientWidth;
        const y_frac = e.offsetY / img.clientHeight;

        const bounds = model.get("axis_bounds");
        if (!bounds) return;

        let hover_plane = null;
        const mpl_y_frac = 1.0 - y_frac;

        for (const [plane, b] of Object.entries(bounds)) {
            const [bx0, by0, bw, bh] = b;
            if (x_frac >= bx0 && x_frac <= bx0 + bw && mpl_y_frac >= by0 && mpl_y_frac <= by0 + bh) {
                hover_plane = plane;
                break;
            }
        }

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

    // We attach keydown to document to catch 'C' presses reliably
    // when hovering over the image, but we only trigger if we have valid hover coords.
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

    // Cleanup listener when widget is destroyed
    model.on("destroy", () => {
        document.removeEventListener("keydown", keydownListener);
    });


    uiTogglesContainer.appendChild(warningSpan);

    // Assemble controlsDiv in requested order
    controlsDiv.appendChild(uiTogglesContainer);
    controlsDiv.appendChild(channelsContainer);
    controlsDiv.appendChild(thicknessContainer);
    controlsDiv.appendChild(positionContainer);
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
