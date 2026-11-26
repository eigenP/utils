// Front end of IsoScatterWidget: the isometric figure, elevation / azimuth
// sliders and a save button. Same conventions as the ortho viewers (D14):
// the dark theme of tnia_plotting_anywidgets.css (colours there, geometry
// inline), labelled controls and a status line for the export outcome (D15).
export default {
  render({ model, el }) {
    el.classList.add("tnia-viewer");
    el.dataset.lmSuppressShortcuts = "true";  // typing in a box is not a notebook command
    el.style.display = "flex";
    el.style.flexDirection = "column";
    el.style.gap = "10px";

    const imgContainer = document.createElement("div");
    imgContainer.style.minWidth = "0";
    const img = document.createElement("img");
    img.alt = "isometric scatter view";
    img.style.maxWidth = "100%";
    img.style.height = "auto";
    img.style.maxHeight = "800px"; // Slightly larger to accommodate 3D + projections
    img.style.display = "block";
    imgContainer.appendChild(img);

    function createSlider(label, traitName, min, max, step) {
      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.alignItems = "center";
      container.style.gap = "10px";

      const labelEl = document.createElement("label");
      labelEl.textContent = label;
      labelEl.style.width = "100px";
      labelEl.style.flex = "none";

      const input = document.createElement("input");
      input.type = "range";
      input.min = min;
      input.max = max;
      input.step = step;
      input.style.flexGrow = "1";
      input.style.minWidth = "0";
      input.setAttribute("aria-label", `${label} (degrees)`);

      const valueDisplay = document.createElement("span");
      valueDisplay.className = "tnia-sub";
      valueDisplay.style.width = "60px";
      valueDisplay.style.textAlign = "right";

      function update() {
        const val = model.get(traitName);
        input.value = val;
        valueDisplay.textContent = `${val}°`;
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

    // Status line: Python's export outcome (status_msg)
    const statusLine = document.createElement("div");
    statusLine.className = "tnia-status";
    statusLine.setAttribute("aria-live", "polite");
    const showStatus = (msg) => {
      statusLine.textContent = msg || "";
      statusLine.classList.toggle("tnia-error", !!msg && msg.startsWith("error"));
    };
    model.on("change:status_msg", () => showStatus(model.get("status_msg")));
    showStatus(model.get("status_msg"));

    // Save controls
    const saveContainer = document.createElement("div");
    saveContainer.style.display = "flex";
    saveContainer.style.flexWrap = "wrap";
    saveContainer.style.gap = "6px 10px";
    saveContainer.style.alignItems = "center";

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

    const saveBtn = document.createElement("button");
    saveBtn.type = "button";
    saveBtn.className = "tnia-btn";
    saveBtn.textContent = "💾 Save";
    saveBtn.title = "Save the figure to the file (format from its extension: svg, pdf, png; never overwrites, adds _1, _2, ...)";
    saveBtn.addEventListener("click", () => {
      showStatus("saving…");
      let current = model.get("save_trigger") || 0;
      model.set("save_trigger", current + 1);
      model.save_changes();
    });

    saveContainer.appendChild(saveLabel);
    saveContainer.appendChild(saveBtn);

    // Assemble UI
    el.appendChild(imgContainer);

    const controlsDiv = document.createElement("div");
    controlsDiv.appendChild(elevSlider);
    controlsDiv.appendChild(azimSlider);

    el.appendChild(controlsDiv);
    el.appendChild(saveContainer);
    el.appendChild(statusLine);

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
