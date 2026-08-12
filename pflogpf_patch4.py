with open("tests/test_single_cell.py", "r") as f:
    content = f.read()

# Remove duplicate line if it got added multiple times
lines = content.splitlines()
out = []
seen = set()
for l in lines:
    if "from unittest.mock import patch, MagicMock" in l:
        if "mock_imports" not in seen:
            seen.add("mock_imports")
            out.append(l)
    else:
        out.append(l)

with open("tests/test_single_cell.py", "w") as f:
    f.write("\n".join(out))
