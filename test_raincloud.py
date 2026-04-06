import pandas as pd
import matplotlib.pyplot as plt
from src.eigenp_utils.plotting_utils import raincloud_plot

df = pd.DataFrame({
    'stage': ['A', 'A', 'B', 'B', 'C', 'C'],
    'distances': [1, 2, 3, 4, 5, 6]
})

fig, ax1 = plt.subplots()

raincloud_plot(x='stage', y='distances',
                hue='stage',
                data=df,
               size_scatter=10,
               size_median=50,
               alpha_scatter=0.2,
               alpha_violin=0.3,
               linewidth_scatter=1,
               linewidth_boxplot=2,
               offset_scatter=0.1,
               ax=ax1)

plt.savefig('test_raincloud.png')
print("Successfully generated plot!")
