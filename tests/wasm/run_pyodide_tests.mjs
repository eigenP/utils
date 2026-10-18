// Run the test suite inside Pyodide, the CPython-on-WebAssembly runtime that
// marimo uses for browser (WASM) notebooks.
//
// Usage (from the repository root, after building a wheel into dist/):
//   npm install --no-save --prefix tests/wasm pyodide@314.0.0
//   node tests/wasm/run_pyodide_tests.mjs
//
// Set PYODIDE_INDEX_URL to a local, fully downloaded Pyodide distribution to
// run offline; otherwise packages are fetched from the Pyodide CDN.
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
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

// Local server for exercising io.download_file_async (pyfetch) without network
const blob = crypto.randomBytes(50_000);
const server = http.createServer((req, res) => {
  if (req.url === "/blob.bin") {
    res.writeHead(200, { "Content-Type": "application/octet-stream" });
    res.end(blob);
  } else {
    res.writeHead(404).end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
pyodide.globals.set("BLOB_URL", `http://127.0.0.1:${server.address().port}/blob.bin`);
pyodide.globals.set("BLOB_SHA256", crypto.createHash("sha256").update(blob).digest("hex"));

const rc = await pyodide.runPythonAsync(
  fs.readFileSync(path.join(here, "pyodide_pytest.py"), "utf8"),
);
server.close();
process.exit(Number(rc));
