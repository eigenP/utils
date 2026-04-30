import numpy as np
import pandas as pd
from scipy.special import gammaln

np.random.seed(42)
N = 1000
clones = np.random.choice([f"C{i}" for i in range(50)], N)
types = np.random.choice([f"T{i}" for i in range(5)], N)

df = pd.DataFrame({"CloneID": clones, "cell_type": types})
binary_matrix = pd.crosstab(df["CloneID"], df["cell_type"]).clip(upper=1)
observed_counts = binary_matrix.T @ binary_matrix

# Permutation
n_permutations = 1000
null_matrices = []
labels_array = np.array(df["cell_type"].astype(str))
for _ in range(n_permutations):
    np.random.shuffle(labels_array)
    shuffled_df = pd.DataFrame({"CloneID": df["CloneID"].values, "cell_type": labels_array})
    shuffled_binary = pd.crosstab(shuffled_df["CloneID"], shuffled_df["cell_type"]).clip(upper=1)
    shuffled_binary = shuffled_binary.reindex(columns=binary_matrix.columns, fill_value=0)
    null_matrices.append((shuffled_binary.T @ shuffled_binary).values)

null_matrices = np.array(null_matrices)
null_mean = null_matrices.mean(axis=0)
null_std = null_matrices.std(axis=0)

print("Permutation Mean T0-T1:", null_mean[0, 1])
print("Permutation Std T0-T1:", null_std[0, 1])

# Analytical
K = df["cell_type"].value_counts().reindex(binary_matrix.columns).values
n_c = df["CloneID"].value_counts().reindex(binary_matrix.index).values

def log_prob_zero(N, K, n):
    K = K[np.newaxis, :]
    n = n[:, np.newaxis]
    K, n = np.broadcast_arrays(K, n)
    valid = (N - K - n) >= 0
    res = np.full_like(K, -np.inf, dtype=float)
    res[valid] = (gammaln(N - K[valid] + 1) + gammaln(N - n[valid] + 1)
                  - gammaln(N - K[valid] - n[valid] + 1) - gammaln(N + 1))
    return np.exp(res)

P_zero_A = log_prob_zero(N, K, n_c)
P_in_A = 1 - P_zero_A
E_diag = P_in_A.sum(axis=0)

K_matrix = K[:, np.newaxis] + K[np.newaxis, :]
K_AB = K_matrix[np.newaxis, :, :]
n_c_3d = n_c[:, np.newaxis, np.newaxis]

K_AB, n_c_3d = np.broadcast_arrays(K_AB, n_c_3d)

valid = (N - K_AB - n_c_3d) >= 0
res = np.full_like(K_AB, -np.inf, dtype=float)
res[valid] = (gammaln(N - K_AB[valid] + 1) + gammaln(N - n_c_3d[valid] + 1)
              - gammaln(N - K_AB[valid] - n_c_3d[valid] + 1) - gammaln(N + 1))
P_zero_AB = np.exp(res)

P_zero_A_3d = P_zero_A[:, :, np.newaxis]
P_zero_B_3d = P_zero_A[:, np.newaxis, :]

P_in_AB = 1 - P_zero_A_3d - P_zero_B_3d + P_zero_AB

E_matrix = P_in_AB.sum(axis=0)
np.fill_diagonal(E_matrix, E_diag)

print("Analytical Mean T0-T1:", E_matrix[0, 1])

Var_matrix = (P_in_AB * (1 - P_in_AB)).sum(axis=0)
np.fill_diagonal(Var_matrix, (P_in_A * (1 - P_in_A)).sum(axis=0))
Std_matrix = np.sqrt(np.maximum(Var_matrix, 0))

print("Analytical Std T0-T1:", Std_matrix[0, 1])
