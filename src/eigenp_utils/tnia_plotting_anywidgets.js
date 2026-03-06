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
    img.style.maxHeight = "600px";
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
    saveBtn.addEventListener("click", () => {
      let current = model.get("save_trigger");
      model.set("save_trigger", current + 1);
      model.save_changes();
    });

    saveContainer.appendChild(saveLabel);
    saveContainer.appendChild(saveInput);
    saveContainer.appendChild(saveBtn);

    el.appendChild(imgContainer);

    const controlsDiv = document.createElement("div");

    // Channels
    const channelNames = model.get("channel_names");
    if (channelNames && channelNames.length > 0) {
      const chanContainer = document.createElement("div");
      chanContainer.style.display = "flex";
      chanContainer.style.flexWrap = "wrap";
      chanContainer.style.gap = "10px";
      chanContainer.style.marginBottom = "10px";

      function updateCheckboxes() {
        const visible = model.get("channel_visible");
        chanContainer.innerHTML = ""; // Clear to rebuild
        channelNames.forEach((name, index) => {
          const label = document.createElement("label");
          label.style.display = "flex";
          label.style.alignItems = "center";
          label.style.gap = "4px";
          label.style.fontSize = "14px";

          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.checked = visible[index];

          cb.addEventListener("change", () => {
             const currentVisible = [...model.get("channel_visible")];
             currentVisible[index] = cb.checked;
             model.set("channel_visible", currentVisible);
             model.save_changes();
          });

          label.appendChild(cb);
          label.appendChild(document.createTextNode(name));
          chanContainer.appendChild(label);
        });
      }

      model.on("change:channel_visible", updateCheckboxes);
      updateCheckboxes();

      controlsDiv.appendChild(chanContainer);
      controlsDiv.appendChild(document.createElement("hr"));
    }

    controlsDiv.appendChild(xThick);
    controlsDiv.appendChild(yThick);
    controlsDiv.appendChild(zThick);
    controlsDiv.appendChild(document.createElement("hr"));
    controlsDiv.appendChild(xPos);
    controlsDiv.appendChild(yPos);
    controlsDiv.appendChild(zPos);
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

    // Annotation Controls (only if annotator widget)
    const hasAnnotation = model.get("annotation_mode") !== undefined;
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
