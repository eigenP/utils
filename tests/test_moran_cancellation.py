import numpy as np
import scipy.sparse as sp
import time

def test_moran_formulas():
    N = 10000
    block = 1024

    # Random sparse X
    np.random.seed(42)
    X = sp.random(N, block, density=0.05, format='csr', dtype=np.float32)

    # Random sparse W (row standardized)
    W = sp.random(N, N, density=0.001, format='csr', dtype=np.float32)
    rs = np.array(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    W = W.multiply(1.0 / rs[:, None]).tocsr()

    # 1. Current formula
    t0 = time.time()
    Xb = X.copy()
    WXb = W @ Xb
    Xb_dense1 = Xb.toarray()
    WXb_dense1 = WXb.toarray()

    mu = Xb_dense1.mean(axis=0)
    W_row_sums = np.array(W.sum(axis=1)).ravel()

    sum_cross = np.einsum('ij,ij->j', Xb_dense1, WXb_dense1)
    sum_sq = np.einsum('ij,ij->j', Xb_dense1, Xb_dense1)
    sum_WXb = WXb_dense1.sum(axis=0)
    sum_xR = Xb_dense1.T @ W_row_sums
    S0 = W.sum()

    num1 = sum_cross - mu * sum_WXb - mu * sum_xR + (mu**2) * S0
    den1 = sum_sq - N * (mu**2)
    t1 = time.time()

    # 2. Proposed formula
    t2 = time.time()
    Xb_dense2 = X.toarray()
    Xb_dense2 -= mu[None, :]
    WXb_dense2 = W @ Xb_dense2

    num2 = np.einsum('ij,ij->j', Xb_dense2, WXb_dense2)
    den2 = np.einsum('ij,ij->j', Xb_dense2, Xb_dense2)
    t3 = time.time()

    print(f"Current: num={num1[:5]}, den={den1[:5]}, time={t1-t0:.4f}s")
    print(f"Proposed: num={num2[:5]}, den={den2[:5]}, time={t3-t2:.4f}s")

    # Check max difference
    print(f"Max diff num: {np.abs(num1 - num2).max()}")
    print(f"Max diff den: {np.abs(den1 - den2).max()}")

if __name__ == '__main__':
    test_moran_formulas()
