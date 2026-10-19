from pathlib import Path
import asyncio
import gzip
import os
import shutil
import sys
import urllib.request
import warnings

# True inside Pyodide (e.g. marimo WASM notebooks), where there are no sockets
IN_WASM = sys.platform == "emscripten"

class FileExistsAssertionError(FileExistsError, AssertionError):
    """Raised by `check_file_before_save`; still an AssertionError for older callers."""


def check_file_before_save(filename, overwrite=False):
    """
    Checks if a file exists before attempting to save.

    Parameters:
    - filename: The path to the file where data will be saved.
    - overwrite: If False and the file exists, raises FileExistsAssertionError
      (a FileExistsError that is also an AssertionError). Unlike a bare assert,
      the check also runs under ``python -O``.
    """
    path = Path(filename)
    print(f'Saving at: {path}')
    if not overwrite and path.exists():
        raise FileExistsAssertionError(f"File exists and overwrite is False: {path}")

def ensure_directory(filepath):
    """
    Ensures that the directory for the given file path exists.
    Creates the directory and any necessary parent directories if they do not exist.

    Parameters:
    - filepath: The path to the file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

def un_gzip(input_file_path, output=None):
    """
    Unzips a .gz file.

    Parameters:
    - input_file_path: The path to the .gz file.
    - output: The path to the output file. If None, removes .gz extension.
    """
    input_path = Path(input_file_path)

    if output is None:
        if input_path.suffix == '.gz':
            output_path = input_path.with_suffix('')
        else:
            output_path = input_path.with_name(input_path.name + '_unzipped')
    else:
        output_path = Path(output)

    # Check if the unzipped file already exists to avoid re-unzipping
    if not output_path.exists():
        print(f"Unzipping {input_path} to {output_path}...")
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Unzipping complete.")
        except Exception as e:
            # Cleanup if failed
            if output_path.exists():
                output_path.unlink()
            raise e
    else:
        print(f"File already unzipped at {output_path}.")

    return output_path

def download_file(url: str, dest: Path, chunk: int = 8192) -> Path:
    """Download url to dest if it doesn't exist yet; returns dest.

    Blocking, so unavailable in the browser (Pyodide); use
    ``await download_file_async(url, dest)`` there.
    """
    dest = Path(dest)
    if dest.exists():
        return dest

    if IN_WASM:
        raise RuntimeError(
            "download_file cannot block in the browser (Pyodide / marimo WASM). "
            "Use `await eigenp_utils.io.download_file_async(url, dest)` instead."
        )

    ensure_directory(dest)
    # Download to a temporary file and move it into place only once it is complete,
    # so an interrupted download is never mistaken for a cached file later
    part = dest.with_name(dest.name + ".part")

    try:
        try:
            # Try using urllib (Standard Library)
            with urllib.request.urlopen(url) as response:
                with open(part, 'wb') as f:
                    shutil.copyfileobj(response, f, length=chunk)
                _check_complete(part, response.headers.get("Content-Length"), url)

        except Exception as e:
            # Fallback to requests
            try:
                import requests
            except ImportError:
                warnings.warn(
                    f"Download with urllib failed: {e}. 'requests' library not found. "
                    "Please install it using 'uv pip install requests' and reload the module: "
                    "import importlib; importlib.reload(eigenp_utils.io)"
                )
                raise e
            print(f"urllib failed ({e}), falling back to requests...")
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(part, "wb") as f:
                for c in resp.iter_content(chunk_size=chunk):
                    if c:
                        f.write(c)
            # Content-Length counts the bytes on the wire; skip the check for encoded bodies
            if not resp.headers.get("Content-Encoding"):
                _check_complete(part, resp.headers.get("Content-Length"), url)

        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()

    return dest


def _check_complete(path: Path, content_length, url: str) -> None:
    """Raise if fewer bytes than the announced Content-Length were written."""
    if content_length is None:
        return
    expected = int(content_length)
    received = path.stat().st_size
    if received != expected:
        raise OSError(
            f"Incomplete download of {url}: received {received} of {expected} bytes."
        )


async def download_file_async(url: str, dest: Path, chunk: int = 8192) -> Path:
    """Download url to dest if it doesn't exist yet; returns dest.

    Works both natively and in the browser (Pyodide / marimo WASM). In the
    browser the request goes through ``pyodide.http.pyfetch``, i.e. the
    browser's ``fetch``, so the server must send CORS headers
    (``Access-Control-Allow-Origin``). Natively it runs ``download_file`` in a
    worker thread.
    """
    dest = Path(dest)
    if not IN_WASM:
        return await asyncio.to_thread(download_file, url, dest, chunk)
    if dest.exists():
        return dest

    from pyodide.http import pyfetch

    try:
        response = await pyfetch(url)
    except Exception as e:
        # Browsers report CORS rejections as a bare network error
        raise OSError(
            f"Fetching {url} failed in the browser ({e}). The server must allow "
            "cross-origin requests (send an Access-Control-Allow-Origin header)."
        ) from e
    if not response.ok:
        raise OSError(f"Fetching {url} failed with HTTP {response.status}.")

    ensure_directory(dest)
    # Write only once the full body has arrived, so a failed download leaves no partial file
    dest.write_bytes(await response.bytes())
    return dest
