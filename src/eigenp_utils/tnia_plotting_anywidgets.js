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
    controlsDiv.appendChild(xThick);
    controlsDiv.appendChild(yThick);
    controlsDiv.appendChild(zThick);
    controlsDiv.appendChild(document.createElement("hr"));
    controlsDiv.appendChild(xPos);
    controlsDiv.appendChild(yPos);
    controlsDiv.appendChild(zPos);

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
