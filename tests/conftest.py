
import pytest
import matplotlib.pyplot as plt
@pytest.fixture(autouse=True)
def auto_close_figures():
    yield
    plt.close('all')
