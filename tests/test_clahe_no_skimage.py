"""clahe_u8 must work without scikit-image (minimal segmentation environments)."""

import subprocess
import sys
import textwrap


def test_clahe_u8_imports_and_runs_without_skimage():
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None  # any `import skimage...` now raises ImportError
        import numpy as np
        from eigenp_utils.clahe_equalize_adapthist import clahe_u8

        img = (np.arange(64 * 64) % 251).reshape(64, 64).astype(np.uint8)
        a = clahe_u8(img, kernel_size=(16, 16), clip_limit=0.01, backend="numpy")
        assert a.dtype == np.uint8 and a.shape == img.shape
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "ok"  # the package prints a hint on import
