with open("tests/test_single_cell.py", "r") as f:
    content = f.read()

content = "from eigenp_utils.single_cell import pflogpf\n" + content

with open("tests/test_single_cell.py", "w") as f:
    f.write(content)
