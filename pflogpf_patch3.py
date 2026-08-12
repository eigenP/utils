with open("tests/test_single_cell.py", "r") as f:
    content = f.read()

# Fix mock import infinite recursion
content = content.replace("""        def side_effect(name, *args, **kwargs):
            if name == "scclr":
                raise ImportError("Mocked ImportError")
            return __import__(name, *args, **kwargs)""",
"""        orig_import = __import__
        def side_effect(name, *args, **kwargs):
            if name == "scclr":
                raise ImportError("Mocked ImportError")
            return orig_import(name, *args, **kwargs)""")

# Add missing mock imports if they aren't there
if "from unittest.mock import patch, MagicMock" not in content:
    content = "from unittest.mock import patch, MagicMock\n" + content

with open("tests/test_single_cell.py", "w") as f:
    f.write(content)
