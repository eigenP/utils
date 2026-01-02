
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import the function from the module
from eigenp_utils.plotting_utils import raincloud_plot

if __name__ == "__main__":
    print("Generating test plots using updated module...")

    # 1. Single Array
    data1 = np.random.normal(10, 2, 100)
    res1 = raincloud_plot(data1, title="Single Array", orientation='vertical')
    res1['fig'].savefig("tests/test_single_module.png")
    plt.close(res1['fig'])

    # 2. DataFrame (Multiple)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(2, 0.5, 100)
    })
    res2 = raincloud_plot(df, title="DataFrame Vertical", palette=["red", "blue"])
    res2['fig'].savefig("tests/test_df_module.png")
    plt.close(res2['fig'])

    # 3. Dict (Horizontal)
    d = {'G1': np.random.randn(50), 'G2': np.random.randn(50) + 3}
    res3 = raincloud_plot(d, title="Dict Horizontal", orientation='horizontal', palette="green")
    res3['fig'].savefig("tests/test_dict_module.png")
    plt.close(res3['fig'])

    # 4. List of Lists (Equal length) - Regression Test
    list_data = [[1, 2, 3], [4, 5, 6]]
    # This should be treated as 2 groups
    res4 = raincloud_plot(list_data, title="List of Lists", palette=["cyan", "magenta"])
    # Verify we have 2 ticks
    n_ticks = len(res4['axes'].get_xticks())
    if n_ticks != 2:
        print(f"FAILURE: Expected 2 ticks for list of lists, got {n_ticks}")
        exit(1)
    else:
        print("SUCCESS: List of lists correctly identified as multiple groups.")
    res4['fig'].savefig("tests/test_list_lists_module.png")
    plt.close(res4['fig'])

    # 5. List of Lists (Jagged)
    jagged_data = [[1, 2, 3], [4, 5, 6, 7]]
    res5 = raincloud_plot(jagged_data, title="Jagged List", palette=["orange", "purple"])
    n_ticks = len(res5['axes'].get_xticks())
    if n_ticks != 2:
        print(f"FAILURE: Expected 2 ticks for jagged list, got {n_ticks}")
        exit(1)
    else:
        print("SUCCESS: Jagged list correctly identified as multiple groups.")
    res5['fig'].savefig("tests/test_jagged_module.png")
    plt.close(res5['fig'])

    print("All module tests passed.")
