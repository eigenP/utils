with open("src/eigenp_utils/tnia_plotting_anywidgets.js", "r") as f:
    js_code = f.read()

# We want to add the canvas under each channel
# We'll locate the channel generation loop and add a canvas element, plus a function to draw it.
# The code is roughly:
#         chDiv.appendChild(createNumberInput("vmin", ...
#         chDiv.appendChild(createNumberInput("vmax", ...
#         chDiv.appendChild(createNumberInput("gamma", ...
#         chDiv.appendChild(createNumberInput("opacity", ...
#         channelsContainer.appendChild(chDiv);

replacement = """        chDiv.appendChild(createNumberInput("gamma", "gamma_list", true, 0, 2.0, false));
        chDiv.appendChild(createNumberInput("opacity", "opacity_list", true, 0, 1, false));

        // Add Histogram Canvas
        const histCanvas = document.createElement("canvas");
        histCanvas.width = 120;
        histCanvas.height = 30;
        histCanvas.style.marginLeft = "auto";
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

          ctx.strokeStyle = color;
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

        channelsContainer.appendChild(chDiv);"""

js_code = js_code.replace("""        chDiv.appendChild(createNumberInput("gamma", "gamma_list", true, 0, 2.0, false));
        chDiv.appendChild(createNumberInput("opacity", "opacity_list", true, 0, 1, false));

        channelsContainer.appendChild(chDiv);""", replacement)


with open("src/eigenp_utils/tnia_plotting_anywidgets.js", "w") as f:
    f.write(js_code)
