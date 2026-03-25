import re

with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()

# Make the median_dist dynamically skip self node distance 0 for better calculation
search_str = """        epsilon = 1e-6
        median_dist = np.median(dist)
        weights = 1.0 / (dist + median_dist + epsilon)"""

replace_str = """        epsilon = 1e-6
        median_dist = np.median(dist[dist > 0]) if np.any(dist > 0) else 0.0
        weights = 1.0 / (dist + median_dist + epsilon)"""

new_content = content.replace(search_str, replace_str)

with open('src/eigenp_utils/single_cell.py', 'w') as f:
    f.write(new_content)
