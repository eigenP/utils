import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dimensionality_parser import parse_slice


def test_parse_slice_single_value():
    assert parse_slice("5") == (5, 6, 1)


def test_parse_slice_range():
    assert parse_slice("1:4") == (1, 4, 1)


def test_parse_slice_step():
    assert parse_slice("2:8:2") == (2, 8, 2)


def test_parse_slice_invalid():
    with pytest.raises(ValueError):
        parse_slice("1:2:3:4")
