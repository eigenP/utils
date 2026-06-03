// Javascript code for the copy params button

        const copyParamsBtn = document.createElement("button");
        copyParamsBtn.innerHTML = "📋"; // copy icon
        copyParamsBtn.title = "Copy parameters to clipboard";
        copyParamsBtn.style.padding = "6px 12px";
        copyParamsBtn.style.backgroundColor = "#e0e0e0";
        copyParamsBtn.style.color = "#333";
        copyParamsBtn.style.border = "1px solid #999";
        copyParamsBtn.style.borderRadius = "4px";
        copyParamsBtn.style.cursor = "pointer";
        copyParamsBtn.style.fontSize = "16px";
        copyParamsBtn.addEventListener("mouseover", () => {
            copyParamsBtn.style.backgroundColor = "#ccc";
        });
        copyParamsBtn.addEventListener("mouseout", () => {
            copyParamsBtn.style.backgroundColor = "#e0e0e0";
        });
        copyParamsBtn.addEventListener("click", () => {
            // Trigger Python to compute the string and set it in a traitlet
            let current = model.get("copy_params_trigger") || 0;
            model.set("copy_params_trigger", current + 1);
            model.save_changes();
        });

        saveContainer.appendChild(copyParamsBtn);

        // Listen for python returning the copy string
        model.on("change:copy_params_string", () => {
            const str = model.get("copy_params_string");
            if (str && navigator.clipboard) {
                navigator.clipboard.writeText(str).then(() => {
                    const originalHTML = copyParamsBtn.innerHTML;
                    copyParamsBtn.innerHTML = "✅";
                    setTimeout(() => {
                        copyParamsBtn.innerHTML = originalHTML;
                    }, 1500);
                }).catch(err => {
                    console.error("Failed to copy text: ", err);
                });
            }
        });
