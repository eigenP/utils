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
    img.style.maxHeight = "800px"; // Slightly larger to accommodate 3D + projections
    imgContainer.appendChild(img);

    function createSlider(label, traitName, min, max, step) {
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.alignItems = "center";
      container.style.gap = "10px";

      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      labelEl.style.width = "100px";

      const input = document.createElement("input");
      input.type = "range";
      input.min = min;
      input.max = max;
      input.step = step;
      input.style.flexGrow = "1";

      const valueDisplay = document.createElement("span");
      valueDisplay.style.width = "60px";
      valueDisplay.style.textAlign = "right";

      function update() {
        const val = model.get(traitName);
        input.value = val;
        valueDisplay.textContent = val;
      }

      update();

      model.on(`change:${traitName}`, update);

      input.addEventListener("input", () => {
        model.set(traitName, parseFloat(input.value));
        model.save_changes();
      });

      container.appendChild(labelEl);
      container.appendChild(input);
      container.appendChild(valueDisplay);

      return container;
    }

    // Camera controls
    // Updated Azimuth range to -180 to 180 to support default -60
    const elevSlider = createSlider("Elevation", "elev", -90, 90, 1);
    const azimSlider = createSlider("Azimuth", "azim", -180, 180, 1);

    // Save controls
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

    // Assemble UI
    el.appendChild(imgContainer);

    const controlsDiv = document.createElement("div");
    controlsDiv.appendChild(elevSlider);
    controlsDiv.appendChild(azimSlider);

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
