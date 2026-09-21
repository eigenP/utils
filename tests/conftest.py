import pytest
import matplotlib.pyplot as plt
import numpy as np

@pytest.fixture(autouse=True)
def auto_close_figures():
    yield
    plt.close('all')

def create_sham_volume(shape=(2, 64, 128, 128), dtype=np.uint8):
    """
    Creates a 2-channel 3D volume (C, Z, Y, X) where:
    - Channel 0: all black except a central 3D cube.
    - Channel 1: all black except a 3D sphere/circle slightly offset from center.
    """
    c, z_dim, y_dim, x_dim = shape
    vol = np.zeros((c, z_dim, y_dim, x_dim), dtype=dtype)
    max_val = 255 if np.issubdtype(dtype, np.integer) else 1.0

    # Channel 0: Cube at center
    cz, cy, cx = z_dim // 2, y_dim // 2, x_dim // 2
    r_cube_z, r_cube_y, r_cube_x = max(1, z_dim // 4), max(1, y_dim // 4), max(1, x_dim // 4)

    z0, z1 = max(0, cz - r_cube_z), min(z_dim, cz + r_cube_z)
    y0, y1 = max(0, cy - r_cube_y), min(y_dim, cy + r_cube_y)
    x0, x1 = max(0, cx - r_cube_x), min(x_dim, cx + r_cube_x)

    vol[0, z0:z1, y0:y1, x0:x1] = max_val

    # Channel 1: Sphere/circle offset from center
    off_z, off_y, off_x = cz + max(1, z_dim // 8), cy + max(1, y_dim // 8), cx + max(1, x_dim // 8)
    r_sphere = max(1, min(z_dim, y_dim, x_dim) // 4)

    gz, gy, gx = np.ogrid[:z_dim, :y_dim, :x_dim]
    mask_sphere = (gz - off_z)**2 + (gy - off_y)**2 + (gx - off_x)**2 <= r_sphere**2
    vol[1, mask_sphere] = max_val

    return vol

@pytest.fixture
def sham_volume():
    """Pytest fixture returning a 2-channel 3D volume of shape (2, 64, 128, 128)."""
    return create_sham_volume()
