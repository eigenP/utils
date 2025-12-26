
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from eigenp_utils.tnia_plotting_3d import show_xyz_max_slabs

# Create mock data (3 channels)
shape = (20, 20, 20)
dapi = np.random.rand(*shape) * 100
fibro = np.random.rand(*shape) * 50
epor = np.random.rand(*shape) * 200

print("Attempting to run show_xyz_max_slabs with list parameters...")

try:
    fig = show_xyz_max_slabs([dapi, fibro, epor],
                       gamma= [1.0, 0.8, 0.7],
                       vmax = [50, 20, 100],
                       vmin = [0, 0, 50])
    print("Success! (Expected)")
except Exception as e:
    print(f"Caught expected exception: {e}")
    # import traceback
    # traceback.print_exc()
