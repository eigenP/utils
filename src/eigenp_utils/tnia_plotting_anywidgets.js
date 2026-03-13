export default {
  render({ model, el }) {
    // Styles
    el.style.display = "flex";
    el.style.flexDirection = "column";
    el.style.gap = "10px";
    el.style.fontFamily = "sans-serif";

    // Create Image Container
    const imgContainer = document.createElement("div");
    const img = document.createElement("img");
    img.style.maxWidth = "100%";
    imgContainer.appendChild(img);

    function createSlider(label, traitName, minTrait, maxTrait, scaleTrait) {
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.alignItems = "center";
      container.style.gap = "10px";

      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      labelEl.style.width = "100px";

      const input = document.createElement("input");
      input.type = "range";
      input.style.flexGrow = "1";

      const valueDisplay = document.createElement("span");
      valueDisplay.style.width = "60px";
      valueDisplay.style.textAlign = "right";

      function update() {
        const val = model.get(traitName);
        const min = model.get(minTrait) || 1;
        const max = model.get(maxTrait);
        const scale = scaleTrait ? (model.get(scaleTrait) || 1.0) : 1.0;

        input.min = min;
        input.max = max;
        input.value = val;

        const displayVal = val * scale;
        // If scale is 1, show integer. If scale is float, show decimals.
        // Actually, just always show 2 decimals if scale is present and not 1?
        // User requested "show corresponding micron values".
        // Let's use toFixed(2) for consistency if we are scaling.
        if (scale !== 1.0) {
            valueDisplay.textContent = displayVal.toFixed(2);
        } else {
            valueDisplay.textContent = val;
        }
      }

      update();

      model.on(`change:${traitName}`, update);
      model.on(`change:${minTrait}`, update);
      model.on(`change:${maxTrait}`, update);
      if (scaleTrait) {
        model.on(`change:${scaleTrait}`, update);
      }

      input.addEventListener("input", () => {
        model.set(traitName, parseInt(input.value));
        model.save_changes();
      });

      container.appendChild(labelEl);
      container.appendChild(input);
      container.appendChild(valueDisplay);

      return container;
    }

    const xThick = createSlider("X Thickness", "x_t", "min_thickness", "x_thick_max", "sxy");
    const yThick = createSlider("Y Thickness", "y_t", "min_thickness", "y_thick_max", "sxy");
    const zThick = createSlider("Z Thickness", "z_t", "min_thickness", "z_thick_max", "sz");

    const xPos = createSlider("X Position", "x_s", "x_min_pos", "x_max_pos", "sxy");
    const yPos = createSlider("Y Position", "y_s", "y_min_pos", "y_max_pos", "sxy");
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

    // Create Layout Split: Left (Sliders) vs Right (Channels)
    const splitContainer = document.createElement("div");
    splitContainer.style.display = "flex";
    splitContainer.style.gap = "20px";
    splitContainer.style.marginBottom = "10px";

    // Left Side: Sliders (60%)
    const slidersContainer = document.createElement("div");
    slidersContainer.style.flex = "60%";
    slidersContainer.style.display = "flex";
    slidersContainer.style.flexDirection = "column";
    slidersContainer.style.gap = "10px";
    slidersContainer.style.paddingRight = "20px";
    slidersContainer.style.borderRight = "1px solid #ccc";

    slidersContainer.appendChild(xThick);
    slidersContainer.appendChild(yThick);
    slidersContainer.appendChild(zThick);
    slidersContainer.appendChild(document.createElement("hr"));
    slidersContainer.appendChild(xPos);
    slidersContainer.appendChild(yPos);
    slidersContainer.appendChild(zPos);

    splitContainer.appendChild(slidersContainer);

    // Right Side: Channels (40%)
    const channelsContainer = document.createElement("div");
    channelsContainer.style.flex = "40%";
    channelsContainer.style.display = "flex";
    channelsContainer.style.flexDirection = "column";
    channelsContainer.style.gap = "10px";
    channelsContainer.style.fontSize = "12px";
    channelsContainer.style.overflowY = "auto";
    channelsContainer.style.maxHeight = "300px";
    channelsContainer.style.paddingLeft = "10px";

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

        channelsContainer.appendChild(chDiv);
      });
    }

    splitContainer.appendChild(channelsContainer);

    controlsDiv.appendChild(splitContainer);
    controlsDiv.appendChild(document.createElement("hr"));

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
    controlsDiv.appendChild(uiTogglesContainer);

    el.appendChild(controlsDiv);
    el.appendChild(saveContainer);

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
