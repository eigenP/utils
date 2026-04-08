import numpy as np
import pytest
import anndata
from eigenp_utils.single_cell import tl_pacmap, get_pacmap_model
from sklearn.datasets import make_blobs, make_swiss_roll

def test_pacmap_reproducibility_blobs():
    """
    Test that a PaCMAP model reconstructed from `adata.uns` native storage
    yields exactly the same `.transform()` output as the original model
    on a structured dataset (Blobs) with 'random' initialization.
    """
    X_blobs, _ = make_blobs(n_samples=500, n_features=20, centers=5, random_state=42)
    X_test = X_blobs[:20] + np.random.normal(0, 0.1, size=(20, X_blobs.shape[1]))

    adata = anndata.AnnData(X=X_blobs)
    adata.obsm["X_pca"] = X_blobs

    # 1. Fit using our wrapper and save the state
    tl_pacmap(adata, n_components=2, n_neighbors=15, random_state=42, save_model=True, init="random")
    reconstructed_model = get_pacmap_model(adata)

    # 2. Fit a baseline identical model natively using pacmap
    import pacmap
    original_model = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=42)
    _ = original_model.fit_transform(X_blobs, init="random")

    # 3. Compare their outputs on unseen test data
    emb_original = original_model.transform(X_test, basis=X_blobs)
    emb_reconstructed = reconstructed_model.transform(X_test, basis=X_blobs)

    np.testing.assert_allclose(
        emb_reconstructed,
        emb_original,
        atol=1e-6,
        err_msg="Reconstructed PaCMAP model yielded different transform results!"
    )

def test_pacmap_reproducibility_swiss_roll():
    """
    Test that a PaCMAP model reconstructed from `adata.uns` native storage
    yields exactly the same `.transform()` output as the original model
    on a structured dataset (Swiss Roll) with 'pca' initialization.
    """
    X_swiss, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    X_test = X_swiss[:20] + np.random.normal(0, 0.1, size=(20, X_swiss.shape[1]))

    adata = anndata.AnnData(X=X_swiss)
    adata.obsm["X_pca"] = X_swiss

    # 1. Fit using our wrapper and save the state
    tl_pacmap(adata, n_components=2, n_neighbors=15, random_state=42, save_model=True, init="pca")
    reconstructed_model = get_pacmap_model(adata)

    # 2. Fit a baseline identical model natively using pacmap
    import pacmap
    original_model = pacmap.PaCMAP(n_components=2, n_neighbors=15, random_state=42)
    _ = original_model.fit_transform(X_swiss, init="pca")

    # 3. Compare their outputs on unseen test data
    emb_original = original_model.transform(X_test, basis=X_swiss)
    emb_reconstructed = reconstructed_model.transform(X_test, basis=X_swiss)

    np.testing.assert_allclose(
        emb_reconstructed,
        emb_original,
        atol=1e-6,
        err_msg="Reconstructed PaCMAP model with PCA init yielded different transform results!"
    )
