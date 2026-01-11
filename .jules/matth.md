
## 2025-05-15 - Archetype Analysis: Magnitude vs. Pattern in Clustering

**Learning:**
Hierarchical clustering using Ward's method on gene expression data (where genes are observations) minimizes the variance of clusters in the input space. If the input space is raw expression values (or even log1p), the Euclidean distance is dominated by the magnitude of expression.
Two genes can be perfectly correlated (identical "pattern") but have vastly different scales (e.g., Gene A in range [0, 1], Gene B in range [0, 1000]). In this case, Ward's method will likely separate them, grouping Gene A with other low-expression genes (even if anti-correlated) and Gene B with other high-expression genes.
This defeats the purpose of finding "Expression Archetypes" (co-expression modules).

**Action:**
To correctly cluster genes by "pattern" or "shape", the gene expression vectors must be standardized (Z-scored per gene) before computing distances/linkage.
This aligns the clustering metric (Euclidean distance on Z-scores) with the desired similarity metric (Pearson Correlation), as $\|z_x - z_y\|^2 = 2n(1 - r_{xy})$.
Future gene clustering implementations must explicitly standardize data if the goal is pattern recognition rather than magnitude grouping.
## 2024-05-24 - PCA vs Mean for Expression Archetypes
**Learning:** The 'mean profile' is a suboptimal estimator for the consensus of a gene module when genes have varying correlation strengths with the latent process. The First Principal Component (PC1) minimizes the squared orthogonal distance and robustly estimates the 'eigengene', effectively downweighting noisy members.
**Action:** Replaced arithmetic mean with PC1 (via SVD) in . Future clustering consensus steps should favor eigenvector-based methods over simple averaging when variables are heteroscedastic.
## 2024-05-24 - PCA vs Mean for Expression Archetypes
**Learning:** The 'mean profile' is a suboptimal estimator for the consensus of a gene module when genes have varying correlation strengths with the latent process. The First Principal Component (PC1) minimizes the squared orthogonal distance and robustly estimates the 'eigengene', effectively downweighting noisy members.
**Action:** Replaced arithmetic mean with PC1 (via SVD) in find_expression_archetypes. Future clustering consensus steps should favor eigenvector-based methods over simple averaging when variables are heteroscedastic.
