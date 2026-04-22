import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Simulate data
np.random.seed(42)
n_cells = 1000
cell_types = np.random.choice(['T', 'B', 'NK', 'Mono'], size=n_cells, p=[0.4, 0.3, 0.2, 0.1])
# create clones of varying sizes
clone_sizes = np.random.lognormal(mean=1, sigma=0.5, size=200).astype(int)
clone_sizes = clone_sizes[clone_sizes > 0]
clone_ids = np.repeat(np.arange(len(clone_sizes)), clone_sizes)

# truncate to n_cells
if len(clone_ids) > n_cells:
    clone_ids = clone_ids[:n_cells]
else:
    clone_ids = np.concatenate([clone_ids, np.random.choice(len(clone_sizes), n_cells - len(clone_ids))])

df = pd.DataFrame({'CloneID': clone_ids, 'cell_type': cell_types})

label_key = "cell_type"
clone_key = "CloneID"

binary_matrix = pd.crosstab(df[clone_key], df[label_key]).clip(upper=1)
observed_counts = binary_matrix.T @ binary_matrix

n_permutations = 1000
null_matrices = []
labels_array = np.array(df[label_key].astype(str))
for _ in range(n_permutations):
    np.random.shuffle(labels_array)
    shuffled_df = pd.DataFrame({clone_key: df[clone_key].values, label_key: labels_array})
    shuffled_binary = pd.crosstab(shuffled_df[clone_key], shuffled_df[label_key]).clip(upper=1)
    shuffled_binary = shuffled_binary.reindex(columns=binary_matrix.columns, fill_value=0)
    null_matrices.append((shuffled_binary.T @ shuffled_binary).values)

null_matrices = np.array(null_matrices)
null_mean = null_matrices.mean(axis=0)
null_std = null_matrices.std(axis=0)

print("Permutation Mean:")
print(null_mean)
print("Permutation Std:")
print(null_std)

# Analytical
N_total = len(df)
type_counts = df[label_key].value_counts()
clone_sizes = df[clone_key].value_counts()

types = binary_matrix.columns
n_types = len(types)
ana_mean = np.zeros((n_types, n_types))
ana_var = np.zeros((n_types, n_types))

for i, t1 in enumerate(types):
    for j, t2 in enumerate(types):
        N1 = type_counts[t1]
        N2 = type_counts[t2]

        for sc in clone_sizes.values:
            if t1 == t2:
                # Expected clones with >= 1 cell of t1
                p1 = 1.0 - hypergeom.pmf(0, N_total, N1, sc)
                p = p1
            else:
                p_no1 = hypergeom.pmf(0, N_total, N1, sc)
                p_no2 = hypergeom.pmf(0, N_total, N2, sc)
                p_no_both = hypergeom.pmf(0, N_total, N1 + N2, sc)
                p = 1.0 - p_no1 - p_no2 + p_no_both
            ana_mean[i, j] += p
            ana_var[i, j] += p * (1 - p)

print("Analytical Mean:")
print(ana_mean)
print("Analytical Std:")
print(np.sqrt(ana_var))
