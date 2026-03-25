import pacmap
import inspect
src = inspect.getsource(pacmap.PaCMAP.sample_pairs)
print("\n".join(src.split("\n")[:30]))
