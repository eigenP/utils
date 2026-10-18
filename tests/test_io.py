import sys
import pytest
from pathlib import Path
from eigenp_utils.io import check_file_before_save, ensure_directory

def test_check_file_before_save(tmp_path):
    """Test that check file before save works as expected."""
    # Test case 1: File does not exist
    test_file = tmp_path / "test_file.txt"
    check_file_before_save(test_file)  # Should not raise

    # Test case 2: File exists, overwrite=False
    test_file.touch()
    with pytest.raises(AssertionError, match="File exists and overwrite is False"):
        check_file_before_save(test_file, overwrite=False)

    # Test case 3: File exists, overwrite=True
    check_file_before_save(test_file, overwrite=True)  # Should not raise

def test_ensure_directory(tmp_path):
    """Test that ensure directory works as expected."""
    # Test case 1: Create nested directory
    nested_file = tmp_path / "subdir1" / "subdir2" / "file.txt"
    ensure_directory(nested_file)
    assert nested_file.parent.exists()
    assert nested_file.parent.is_dir()

    # Test case 2: Directory already exists
    ensure_directory(nested_file)  # Should not raise


def _serve_directory(directory):
    """Start a background HTTP server for ``directory``; returns (server, base_url)."""
    import functools
    import http.server
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _run_coroutine(coro):
    """Run ``coro`` to completion on a fresh event loop in a worker thread.

    ``asyncio.run`` refuses to start when the calling thread already runs a loop,
    which pytest-playwright's sync API does for the rest of the session once a
    frontend test has used it.
    """
    import asyncio
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@pytest.mark.skipif(sys.platform == "emscripten", reason="no sockets / HTTP server in Pyodide")
def test_download_file_async_native_roundtrip(tmp_path):
    """download_file_async fetches bytes exactly and skips existing files.

    Strategy: serve random binary bytes from a local HTTP server (no external
    network), download them with the async API, and compare byte-for-byte.
    Binary content guards against text decoding on the transport path. Then
    overwrite the destination and download again: an existing file must be left
    untouched, matching download_file's cache semantics.

    Expected outcome: the first call writes identical bytes (in a newly created
    sub-directory); the second returns the path without re-downloading.
    """
    import os
    from eigenp_utils.io import download_file_async

    src = tmp_path / "srv"
    src.mkdir()
    payload = os.urandom(50_000)
    (src / "blob.bin").write_bytes(payload)
    server, base = _serve_directory(src)
    try:
        dest = tmp_path / "out" / "blob.bin"
        result = _run_coroutine(download_file_async(f"{base}/blob.bin", dest))
        assert Path(result) == dest
        assert dest.read_bytes() == payload

        dest.write_bytes(b"cached")
        _run_coroutine(download_file_async(f"{base}/blob.bin", dest))
        assert dest.read_bytes() == b"cached"
    finally:
        server.shutdown()


def test_download_file_refuses_to_block_in_wasm(tmp_path, monkeypatch):
    """In Pyodide, download_file raises a clear error pointing to the async API.

    urllib has no sockets in the browser and the failure would otherwise surface
    as an opaque URLError. Strategy: force the module's IN_WASM flag and call
    download_file for a missing destination. Expected outcome: a RuntimeError
    naming download_file_async, and no file created.
    """
    import eigenp_utils.io as eio

    monkeypatch.setattr(eio, "IN_WASM", True)
    dest = tmp_path / "x.bin"
    with pytest.raises(RuntimeError, match="download_file_async"):
        eio.download_file("http://127.0.0.1:9/never", dest)
    assert not dest.exists()
