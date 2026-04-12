import marimo
import sys
import subprocess
from pathlib import Path

notebook_path = "notebooks/demo_intensity_rescaling.py"
print(f"Running marimo notebook: {notebook_path}")

try:
    subprocess.run(["marimo", "run", notebook_path, "--host", "0.0.0.0", "--port", "8000"], check=True)
except Exception as e:
    print(f"Failed to run notebook: {e}")
