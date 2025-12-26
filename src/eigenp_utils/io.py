from pathlib import Path

def check_file_before_save(filename, overwrite=False):
    """
    Checks if a file exists before attempting to save. Raises an AssertionError if conditions are not met.

    Parameters:
    - filename: The path to the file where data will be saved.
    - overwrite: If False and the file exists, raises an AssertionError.
    """
    path = Path(filename)
    print(f'Saving at: {path}')
    assert overwrite or not path.exists(), "File exists and overwrite is False."

def ensure_directory(filepath):
    """
    Ensures that the directory for the given file path exists.
    Creates the directory and any necessary parent directories if they do not exist.

    Parameters:
    - filepath: The path to the file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
