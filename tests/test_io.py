"""Tests for ``eigenp_utils.io`` filesystem, gzip, and native/WASM download utilities."""
import asyncio
import concurrent.futures
import contextlib
import functools
import gzip
import http.client
import http.server
import os
import subprocess
import sys
import threading
import types
import urllib.request
from pathlib import Path

import pytest

import eigenp_utils.io as eio
from eigenp_utils.io import (
    FileExistsAssertionError,
    check_file_before_save,
    download_file,
    download_file_async,
    ensure_directory,
    un_gzip,
)


@contextlib.contextmanager
def _serve_http(handler_cls):
    """Run a local ThreadingHTTPServer for ``handler_cls`` and close its socket on exit."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _serve_directory(directory):
    """Serve static files from ``directory`` on an ephemeral localhost port."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a: None
    with _serve_http(handler) as base_url:
        yield base_url


def _run_coroutine(coro):
    """Run ``coro`` to completion on a fresh event loop in a worker thread.

    ``asyncio.run`` refuses to start when the calling thread already runs a loop,
    which pytest-playwright's sync API does for the rest of the session once a
    frontend test has used it.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# File & directory guards
# ---------------------------------------------------------------------------


def test_check_file_before_save(tmp_path):
    """Verify overwrite protection and dual exception inheritance for Path and str inputs.

    Strategy: call ``check_file_before_save`` on a missing file, an existing file
    with ``overwrite=False`` (as both ``Path`` and ``str``), and an existing file
    with ``overwrite=True``.
    Expected outcome: missing files and ``overwrite=True`` pass; existing files
    with ``overwrite=False`` raise ``FileExistsAssertionError``, which satisfies
    both ``FileExistsError`` and legacy ``AssertionError`` handlers.
    """
    test_file = tmp_path / "test_file.txt"
    check_file_before_save(test_file)

    test_file.touch()
    for target in (test_file, str(test_file)):
        with pytest.raises(FileExistsAssertionError, match="File exists and overwrite is False") as exc_info:
            check_file_before_save(target, overwrite=False)
        assert isinstance(exc_info.value, FileExistsError)
        assert isinstance(exc_info.value, AssertionError)

    check_file_before_save(test_file, overwrite=True)


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide")
def test_check_file_before_save_enforced_under_python_O(tmp_path):
    """Verify ``check_file_before_save`` still raises under ``python -O``.

    Strategy: invoke a fresh ``python -O`` subprocess calling
    ``check_file_before_save`` on an existing file with ``overwrite=False``.
    Expected outcome: raises ``FileExistsError`` rather than being stripped out
    like a bare ``assert`` statement.
    """
    existing = tmp_path / "exists.txt"
    existing.write_text("x")

    code = (
        "import sys\n"
        "from eigenp_utils.io import check_file_before_save\n"
        "try:\n"
        "    check_file_before_save(sys.argv[1], overwrite=False)\n"
        "except FileExistsError:\n"
        "    print('raised')\n"
    )
    out = subprocess.run(
        [sys.executable, "-O", "-c", code, str(existing)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "raised"


def test_ensure_directory(tmp_path):
    """Verify ``ensure_directory`` creates parent directories without creating the file itself.

    Strategy: call ``ensure_directory`` on a deeply nested non-existent file path
    (both ``Path`` and ``str``), then call it again once the directory exists.
    Expected outcome: all parent directories are created, the target file itself
    does not exist, and repeated calls are idempotent.
    """
    nested_file = tmp_path / "subdir1" / "subdir2" / "file.txt"
    ensure_directory(nested_file)
    assert nested_file.parent.is_dir()
    assert not nested_file.exists()

    # Idempotent when directory already exists (also accepts str)
    ensure_directory(str(nested_file))
    assert nested_file.parent.is_dir()
    assert not nested_file.exists()


# ---------------------------------------------------------------------------
# Gzip decompression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_name, explicit_out, expected_name",
    [
        ("archive.bin.gz", None, "archive.bin"),
        ("archive.dat", None, "archive.dat_unzipped"),
        ("archive.bin.gz", "custom_out.bin", "custom_out.bin"),
    ],
)
def test_un_gzip_roundtrip_and_naming(tmp_path, src_name, explicit_out, expected_name):
    """Verify binary round-trip decompression and output path naming conventions.

    Strategy: compress random binary bytes into a gzip file with ``.gz`` and
    non-``.gz`` names, then decompress via ``un_gzip`` with default and explicit
    output paths.
    Expected outcome: returns the expected ``Path`` and writes byte-identical
    binary contents.
    """
    payload = bytes(range(256)) * 200
    gz_path = tmp_path / src_name
    with gzip.open(gz_path, "wb") as f:
        f.write(payload)

    out_arg = (tmp_path / explicit_out) if explicit_out else None
    result = un_gzip(gz_path, output=out_arg)

    expected_path = tmp_path / expected_name
    assert result == expected_path
    assert expected_path.read_bytes() == payload


def test_un_gzip_cache_and_failure_cleanup(tmp_path):
    """Verify ``un_gzip`` skips existing outputs and cleans up partial files on error.

    Strategy: (1) pre-populate the expected output path and call ``un_gzip`` on a
    valid archive; (2) pass a truncated/corrupt gzip file; (3) pass a missing file.
    Expected outcome: (1) cached output is untouched; (2) corrupt archive raises
    and removes any partially written output file; (3) missing input raises
    ``FileNotFoundError``.
    """
    gz_path = tmp_path / "data.txt.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(b"fresh payload")

    cached_out = tmp_path / "data.txt"
    cached_out.write_bytes(b"already cached")
    assert un_gzip(gz_path) == cached_out
    assert cached_out.read_bytes() == b"already cached"

    # Corrupt archive must raise and leave no partial output file behind
    corrupt_gz = tmp_path / "corrupt.bin.gz"
    corrupt_gz.write_bytes(b"not a valid gzip stream")
    corrupt_out = tmp_path / "corrupt.bin"
    with pytest.raises((OSError, EOFError, gzip.BadGzipFile)):
        un_gzip(corrupt_gz)
    assert not corrupt_out.exists()

    with pytest.raises(FileNotFoundError):
        un_gzip(tmp_path / "does_not_exist.gz")


# ---------------------------------------------------------------------------
# Downloads (native sync, fallback, truncation, async native & WASM)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "emscripten", reason="no sockets / HTTP server in Pyodide")
def test_download_file_sync_roundtrip_cache_and_404(tmp_path, monkeypatch):
    """Verify synchronous ``download_file`` round-trip, cache short-circuit, and HTTP error cleanup.

    Strategy: serve a random binary blob from a local HTTP server; download it to
    a nested path; overwrite the cached file and re-download (including with
    ``IN_WASM=True`` to prove cache hits return before the WASM guard); then
    request a missing URL (HTTP 404).
    Expected outcome: first call creates parent dirs and writes exact bytes;
    cached destination is left untouched even when ``IN_WASM=True``; 404 raises
    and leaves neither destination nor ``.part`` files behind.
    """
    src = tmp_path / "srv"
    src.mkdir()
    payload = os.urandom(32_000)
    (src / "blob.bin").write_bytes(payload)

    with _serve_directory(src) as base:
        dest = tmp_path / "nested" / "dir" / "blob.bin"
        result = download_file(f"{base}/blob.bin", dest)
        assert Path(result) == dest
        assert dest.read_bytes() == payload
        assert not dest.with_name("blob.bin.part").exists()

        # Cache hit returns immediately, even in WASM mode
        dest.write_bytes(b"cached")
        monkeypatch.setattr(eio, "IN_WASM", True)
        assert download_file(f"{base}/blob.bin", dest) == dest
        assert dest.read_bytes() == b"cached"
        monkeypatch.setattr(eio, "IN_WASM", False)

        # HTTP 404 must fail cleanly without leaving dest or .part
        missing_dest = tmp_path / "missing_out" / "nope.bin"
        with pytest.raises(OSError):
            download_file(f"{base}/does_not_exist.bin", missing_dest)
        assert not missing_dest.exists()
        assert not missing_dest.with_name("nope.bin.part").exists()


@pytest.mark.skipif(sys.platform == "emscripten", reason="no sockets / HTTP server in Pyodide")
def test_download_file_rejects_truncated_body(tmp_path):
    """Verify a connection dropped before ``Content-Length`` bytes raises and caches nothing.

    Strategy: serve a response announcing ``Content-Length: 100000`` that closes
    after sending only 1000 bytes, then call ``download_file``.
    Expected outcome: raises a network/IO error and leaves neither the destination
    file nor any ``.part`` staging file in ``tmp_path``.
    """
    class Truncating(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"x" * 1000)
            self.wfile.flush()
            self.close_connection = True

        def log_message(self, *args):
            pass

    dest = tmp_path / "blob.bin"
    with _serve_http(Truncating) as base:
        with pytest.raises((OSError, http.client.HTTPException)):
            download_file(f"{base}/blob.bin", dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_file_requests_fallback(tmp_path, monkeypatch):
    """Verify ``download_file`` falls back to ``requests`` and handles Content-Encoding or absence.

    Strategy: make ``urllib.request.urlopen`` raise ``OSError``; (1) inject a fake
    ``requests`` module returning decoded bytes with ``Content-Encoding: gzip`` and
    a smaller wire ``Content-Length`` to verify the completeness check is skipped
    for encoded bodies; (2) block ``requests`` import via ``sys.modules['requests'] = None``.
    Expected outcome: (1) succeeds and writes exact bytes with no leftover ``.part``;
    (2) emits a ``UserWarning`` mentioning ``requests``, re-raises the original
    ``urllib`` error, and leaves no files behind.
    """
    monkeypatch.setattr(eio, "IN_WASM", False)

    def fail_urlopen(*args, **kwargs):
        raise OSError("simulated urllib failure")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    # Case 1: requests fallback succeeds even when Content-Encoding makes wire length != decoded length
    decoded_body = b"decoded content longer than wire length"

    class FakeResponse:
        headers = {"Content-Encoding": "gzip", "Content-Length": "10"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=8192):
            yield decoded_body[:12]
            yield b""
            yield decoded_body[12:]

    fake_requests = types.SimpleNamespace(get=lambda url, stream=True, timeout=30: FakeResponse())
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    dest = tmp_path / "fallback" / "ok.bin"
    assert download_file("http://example.invalid/ok.bin", dest) == dest
    assert dest.read_bytes() == decoded_body
    assert not dest.with_name("ok.bin.part").exists()

    # Case 2: requests unavailable -> warns and re-raises urllib error with no leftovers
    monkeypatch.setitem(sys.modules, "requests", None)
    dest_fail = tmp_path / "no_req" / "fail.bin"
    with pytest.warns(UserWarning, match="requests"):
        with pytest.raises(OSError, match="simulated urllib failure"):
            download_file("http://example.invalid/fail.bin", dest_fail)
    assert not dest_fail.exists()
    assert not dest_fail.with_name("fail.bin.part").exists()


def test_download_file_refuses_to_block_in_wasm(tmp_path, monkeypatch):
    """In Pyodide, ``download_file`` raises a clear error pointing to ``download_file_async``.

    Strategy: force ``IN_WASM = True`` and call ``download_file`` for a missing
    destination.
    Expected outcome: raises ``RuntimeError`` naming ``download_file_async`` and
    creates no file.
    """
    monkeypatch.setattr(eio, "IN_WASM", True)
    dest = tmp_path / "x.bin"
    with pytest.raises(RuntimeError, match="download_file_async"):
        download_file("http://127.0.0.1:9/never", dest)
    assert not dest.exists()


@pytest.mark.skipif(sys.platform == "emscripten", reason="no sockets / HTTP server in Pyodide")
def test_download_file_async_native_roundtrip(tmp_path):
    """``download_file_async`` fetches bytes exactly and skips existing files natively.

    Strategy: serve random binary bytes from a local HTTP server, download them
    with the async API, compare byte-for-byte, then overwrite the destination
    and re-download.
    Expected outcome: first call writes identical bytes into a newly created
    sub-directory; second call preserves cached content without re-downloading.
    """
    src = tmp_path / "srv"
    src.mkdir()
    payload = os.urandom(50_000)
    (src / "blob.bin").write_bytes(payload)

    with _serve_directory(src) as base:
        dest = tmp_path / "out" / "blob.bin"
        result = _run_coroutine(download_file_async(f"{base}/blob.bin", dest))
        assert Path(result) == dest
        assert dest.read_bytes() == payload

        dest.write_bytes(b"cached")
        _run_coroutine(download_file_async(f"{base}/blob.bin", dest))
        assert dest.read_bytes() == b"cached"


def _drive_sync_coroutine(coro):
    """Step a coroutine whose awaits resolve immediately (works in Pyodide without threads)."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    coro.close()
    raise AssertionError("Coroutine unexpectedly yielded an awaited future")


def test_download_file_async_wasm_pyfetch_paths(tmp_path, monkeypatch):
    """Verify ``download_file_async`` WASM branch: success, cache hit, CORS error, and HTTP error.

    Strategy: set ``IN_WASM = True`` and inject a fake ``pyodide.http.pyfetch``
    covering (1) successful binary download into a new directory, (2) cache hit
    skipping ``pyfetch``, (3) network/CORS exception wrapping, and (4) non-OK
    HTTP status error.
    Expected outcome: (1) exact bytes written; (2) ``pyfetch`` not called on
    existing file; (3) ``OSError`` mentions ``Access-Control-Allow-Origin``;
    (4) ``OSError`` includes HTTP status code and writes no file.
    """
    monkeypatch.setattr(eio, "IN_WASM", True)
    calls = []
    mode = {"behavior": "ok"}
    payload = b"\x00\x01\x02wasm-binary-payload\xff"

    class FakePyfetchResp:
        def __init__(self, ok, status, data):
            self.ok = ok
            self.status = status
            self._data = data

        async def bytes(self):
            return self._data

    async def fake_pyfetch(url):
        calls.append(url)
        if mode["behavior"] == "ok":
            return FakePyfetchResp(True, 200, payload)
        if mode["behavior"] == "cors":
            raise RuntimeError("TypeError: Failed to fetch")
        return FakePyfetchResp(False, 404, b"")

    if "pyodide.http" in sys.modules:
        monkeypatch.setattr(sys.modules["pyodide.http"], "pyfetch", fake_pyfetch)
    else:
        pyodide_http = types.ModuleType("pyodide.http")
        pyodide_http.pyfetch = fake_pyfetch
        pyodide_pkg = types.ModuleType("pyodide")
        pyodide_pkg.http = pyodide_http
        monkeypatch.setitem(sys.modules, "pyodide", pyodide_pkg)
        monkeypatch.setitem(sys.modules, "pyodide.http", pyodide_http)

    # 1. Success + parent dir creation
    dest = tmp_path / "wasm_out" / "blob.bin"
    result = _drive_sync_coroutine(download_file_async("https://example.invalid/blob.bin", dest))
    assert result == dest
    assert dest.read_bytes() == payload
    assert len(calls) == 1

    # 2. Cache hit must not call pyfetch
    dest.write_bytes(b"cached")
    assert _drive_sync_coroutine(download_file_async("https://example.invalid/blob.bin", dest)) == dest
    assert dest.read_bytes() == b"cached"
    assert len(calls) == 1

    # 3. Network/CORS rejection -> OSError with CORS guidance
    mode["behavior"] = "cors"
    cors_dest = tmp_path / "cors.bin"
    with pytest.raises(OSError, match="Access-Control-Allow-Origin"):
        _drive_sync_coroutine(download_file_async("https://example.invalid/cors.bin", cors_dest))
    assert not cors_dest.exists()

    # 4. Non-OK HTTP status -> OSError with status code and no file created
    mode["behavior"] = "404"
    err_dest = tmp_path / "missing.bin"
    with pytest.raises(OSError, match="HTTP 404"):
        _drive_sync_coroutine(download_file_async("https://example.invalid/missing.bin", err_dest))
    assert not err_dest.exists()
