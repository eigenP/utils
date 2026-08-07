import numpy as np

C = 2
for vmin in [None, 0.0, [None, None], [0.0, 1.0]]:
    res = [None] * C if vmin is None else np.broadcast_to(vmin, (C,)).tolist()
    print(f"vmin={vmin} -> {res}")
