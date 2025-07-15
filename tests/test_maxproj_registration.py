import sys
import types
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Provide a stub pandas module so the import succeeds
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

from maxproj_registration import zero_shift_multi_dimensional


def test_zero_shift_positive_negative():
    arr = np.arange(9).reshape(3, 3)
    result = zero_shift_multi_dimensional(arr, shifts=(1, -1), fill_value=-1)
    expected = np.array([
        [-1, -1, -1],
        [1, 2, -1],
        [4, 5, -1],
    ])
    assert np.array_equal(result, expected)


def test_zero_shift_errors():
    arr = np.zeros((2, 2))
    with pytest.raises(ValueError):
        zero_shift_multi_dimensional(arr, shifts=(1,))
    with pytest.raises(TypeError):
        zero_shift_multi_dimensional(arr, shifts=(1.0, 2.0))
