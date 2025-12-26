import pytest
from pathlib import Path
from eigenp_utils.io import check_file_before_save, ensure_directory

def test_check_file_before_save(tmp_path):
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
    # Test case 1: Create nested directory
    nested_file = tmp_path / "subdir1" / "subdir2" / "file.txt"
    ensure_directory(nested_file)
    assert nested_file.parent.exists()
    assert nested_file.parent.is_dir()

    # Test case 2: Directory already exists
    ensure_directory(nested_file)  # Should not raise
