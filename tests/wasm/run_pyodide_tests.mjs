// Run the test suite inside Pyodide, the CPython-on-WebAssembly runtime that
// marimo uses for browser (WASM) notebooks.
//
// Usage (from the repository root, after building a wheel into dist/):
//   npm install --no-save --prefix tests/wasm pyodide@314.0.0
//   node tests/wasm/run_pyodide_tests.mjs
//
// Set PYODIDE_INDEX_URL to a local, fully downloaded Pyodide distribution to
// run offline; otherwise packages are fetched from the Pyodide CDN.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../..");
const indexURL = process.env.PYODIDE_INDEX_URL;

const { loadPyodide } = indexURL
  ? await import(pathToFileURL(path.join(indexURL, "pyodide.mjs")).href)
  : await import(pathToFileURL(path.join(here, "node_modules/pyodide/pyodide.mjs")).href);
const pyodide = await loadPyodide(indexURL ? { indexURL } : {});

pyodide.FS.mkdirTree("/repo");
pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: repo }, "/repo");
await pyodide.loadPackage("micropip");

const wheel = fs.readdirSync(path.join(repo, "dist")).find((f) => f.endsWith(".whl"));
if (!wheel) throw new Error("No wheel in dist/; run `uv build --wheel` first.");
pyodide.globals.set("WHEEL", `emfs:/repo/dist/${wheel}`);

const rc = await pyodide.runPythonAsync(
  fs.readFileSync(path.join(here, "pyodide_pytest.py"), "utf8"),
);
process.exit(Number(rc));
