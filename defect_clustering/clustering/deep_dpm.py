"""
Lightweight DeepDPM -- Non-Parametric Deep Clustering
=====================================================

Implements the core algorithm from:
    Ronen, Finder & Freifeld (CVPR 2022)
    "DeepDPM: Deep Clustering With an Unknown Number of Clusters"

Instead of requiring the full DeepDPM codebase (PyTorch Lightning 1.2.10),
this module provides a **standalone, sklearn-compatible** implementation
that captures the key ideas:

1. **Bayesian GMM initialisation** -- uses sklearn's BayesianGaussianMixture
   (variational Dirichlet Process) to get an initial K and soft assignments.

2. **MLP cluster network** -- a small PyTorch MLP trained to predict soft
   cluster assignments, using KL divergence against a GMM-derived target.

3. **Split / Merge via Hastings ratio** -- periodically evaluates whether
   splitting or merging clusters improves the Bayesian marginal likelihood
   (Normal-Inverse-Wishart prior, same as the original DeepDPM).

4. **Automatic K discovery** -- starts from an initial guess and converges
   to the data-supported number of clusters through split/merge dynamics.

Key difference from the original:
    - No PyTorch Lightning dependency -- uses raw PyTorch training loop.
    - No Neptune logging -- prints to stdout.
    - Operates directly on pre-extracted features (numpy arrays).
    - Simpler hyperparameter set (works out-of-the-box for 10-200 dim).

Usage:
    from defect_clustering.clustering.deep_dpm import DeepDPMClustering

    model = DeepDPMClustering(init_k=5, max_epochs=200)
    labels = model.fit_predict(features)  # (N, D) numpy array
    print(f"Found {model.n_clusters_} clusters")

References:
    Ronen et al. (2022) -- DeepDPM (CVPR 2022), arXiv:2203.14309
    Blei & Jordan (2006) -- Variational inference for Dirichlet process mixtures
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np


class DeepDPMClustering:
    """Non-parametric deep clustering with automatic K discovery.

    Combines a variational Dirichlet Process GMM with an optional MLP
    refinement network and Bayesian split/merge operations.

    Parameters
    ----------
    init_k : int
        Initial guess for the number of clusters.  The algorithm will
        grow or shrink K from this starting point.
    max_epochs : int
        Number of MLP training epochs (set to 0 for pure DPGMM mode).
    split_merge_every : int
        Evaluate split/merge proposals every N epochs.
    start_split_merge : int
        Epoch at which to begin proposing splits and merges.
    alpha : float
        Dirichlet Process concentration parameter.  Higher values
        encourage more clusters.
    hidden_dims : list[int]
        Hidden layer sizes for the MLP cluster network.
    lr : float
        Learning rate for the MLP.
    batch_size : int
        Mini-batch size for MLP training.
    min_cluster_size : int
        Minimum number of points to form a cluster.  Clusters smaller
        than this after a split are rejected.
    random_state : int
        Random seed for reproducibility.
    use_mlp : bool
        If True, train an MLP cluster network for refined assignments.
        If False, use only the DPGMM (faster, still non-parametric).
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        init_k: int = 10,
        max_epochs: int = 150,
        split_merge_every: int = 25,
        start_split_merge: int = 40,
        alpha: float = 10.0,
        hidden_dims: Optional[list] = None,
        lr: float = 0.002,
        batch_size: int = 256,
        min_cluster_size: int = 10,
        random_state: int = 42,
        use_mlp: bool = True,
        verbose: bool = True,
    ) -> None:
        self.init_k = init_k
        self.max_epochs = max_epochs
        self.split_merge_every = split_merge_every
        self.start_split_merge = start_split_merge
        self.alpha = alpha
        self.hidden_dims = hidden_dims or [128, 64]
        self.lr = lr
        self.batch_size = batch_size
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        self.use_mlp = use_mlp
        self.verbose = verbose

        # Fitted attributes
        self.n_clusters_: int = 0
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.split_merge_history_: list = []
        self._dpgmm = None
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "DeepDPMClustering":
        """Fit the model, discovering K automatically.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Feature matrix (pre-extracted CNN embeddings or fused features).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        N, D = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.verbose:
            print(f"[DeepDPM] Input: {N} samples, {D} dimensions")
            print(f"[DeepDPM] Initial K = {self.init_k}, alpha = {self.alpha}")

        # ── Phase 1: Bayesian GMM (Dirichlet Process) ────────────────────
        self._dpgmm = self._fit_dpgmm(X)
        labels = self._dpgmm.predict(X)

        # Prune inactive components: keep only those with weight > 1/max_comp
        # (the DP prior assigns near-zero weight to unused components)
        weights = self._dpgmm.weights_
        active_mask = weights > (1.0 / len(weights))
        active_components = np.where(active_mask)[0]

        if len(active_components) < len(weights):
            # Remap labels to only active components
            remap = {}
            new_id = 0
            for comp_id in active_components:
                remap[comp_id] = new_id
                new_id += 1
            # Points assigned to pruned components → nearest active component
            active_means = self._dpgmm.means_[active_components]
            for i in range(N):
                if labels[i] in remap:
                    labels[i] = remap[labels[i]]
                else:
                    # Find nearest active component
                    dists = np.linalg.norm(X[i] - active_means, axis=1)
                    labels[i] = dists.argmin()

        K = len(np.unique(labels))
        centers = self._compute_centers(X, labels)

        if self.verbose:
            print(f"[DeepDPM] DPGMM initial: K = {K} active components "
                  f"(pruned from {len(weights)})")

        # ── Phase 2: MLP refinement with split/merge ─────────────────────
        if self.use_mlp and self.max_epochs > 0:
            labels, centers, K = self._mlp_refine(X, labels, centers, K, rng)
        else:
            # Pure DPGMM mode -- apply split/merge on GMM parameters
            labels, centers, K = self._parametric_split_merge(
                X, labels, centers, K, rng
            )

        self.n_clusters_ = K
        self.labels_ = labels
        self.cluster_centers_ = centers
        self._fitted = True

        if self.verbose:
            print(f"[DeepDPM] Final: K = {K} clusters")
            sizes = np.bincount(labels, minlength=K)
            print(f"[DeepDPM] Cluster sizes: {dict(enumerate(sizes.tolist()))}")

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to nearest cluster center."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2,
        )
        return dists.argmin(axis=1)

    # ------------------------------------------------------------------
    # Phase 1: Dirichlet Process GMM via sklearn
    # ------------------------------------------------------------------

    def _fit_dpgmm(self, X: np.ndarray):
        """Fit a variational Bayesian GMM (Dirichlet Process approximation).

        Uses weight_concentration_prior_type='dirichlet_process' which
        implements the stick-breaking construction (Blei & Jordan 2006).
        Components with negligible weight are automatically pruned.
        """
        from sklearn.mixture import BayesianGaussianMixture

        # Allow up to 2x init_k components -- the DP prior will prune
        max_components = max(self.init_k * 2, 50)

        # Use alpha/max_components as concentration prior so that the
        # expected number of active components ≈ alpha * log(N)
        # For alpha=10, N=20000: ~10*10 ≈ 100 (upper bound, DP prunes)
        dpgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=self.alpha / max_components,
            max_iter=500,
            n_init=3,
            tol=1e-4,
            random_state=self.random_state,
            verbose=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dpgmm.fit(X)

        return dpgmm

    # ------------------------------------------------------------------
    # Phase 2: MLP refinement with split/merge (PyTorch)
    # ------------------------------------------------------------------

    def _mlp_refine(
        self, X: np.ndarray, labels: np.ndarray,
        centers: np.ndarray, K: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Train MLP cluster network with periodic split/merge."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            if self.verbose:
                print("[DeepDPM] PyTorch not available -- falling back to "
                      "parametric split/merge (no MLP)")
            return self._parametric_split_merge(X, labels, centers, K, rng)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose and device.type == "cuda":
            print(f"[DeepDPM] Using GPU: {torch.cuda.get_device_name(0)}")
        N, D = X.shape
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        # Build MLP: input_dim -> hidden -> K
        net = self._build_mlp(D, K).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Initial soft targets from current labels (one-hot smoothed)
        targets = self._labels_to_soft(labels, K)

        for epoch in range(self.max_epochs):
            # Mini-batch training
            perm = torch.randperm(N, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, self.batch_size):
                idx = perm[i : i + self.batch_size]
                x_batch = X_t[idx]
                t_batch = torch.tensor(
                    targets[idx.cpu().numpy()],
                    dtype=torch.float32, device=device,
                )

                logits = net(x_batch)
                log_probs = torch.log_softmax(logits, dim=1)
                # KL divergence: sum t * log(t/p) = sum t * (log_t - log_p)
                t_log = torch.log(t_batch + 1e-12)
                loss = (t_batch * (t_log - log_probs)).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Update labels from MLP predictions
            with torch.no_grad():
                logits_all = net(X_t)
                probs = torch.softmax(logits_all, dim=1).cpu().numpy()
                labels = probs.argmax(axis=1)

            # Split / Merge proposals
            if (epoch >= self.start_split_merge and
                    epoch % self.split_merge_every == 0):

                labels_new, centers_new, K_new, action = self._propose_split_merge(
                    X, labels, K, rng
                )
                if K_new != K:
                    self.split_merge_history_.append({
                        "epoch": epoch,
                        "action": action,
                        "k_before": K,
                        "k_after": K_new,
                    })
                    K = K_new
                    labels = labels_new
                    centers = centers_new

                    # Explicitly release GPU memory held by the old MLP + optimizer
                    # before rebuilding — otherwise torch.cuda retains unreferenced
                    # graphs and fragments memory across many split/merge cycles.
                    del net, optimizer, targets
                    if device.type == "cuda":
                        import torch as _torch
                        _torch.cuda.empty_cache()
                    import gc as _gc
                    _gc.collect()

                    targets = self._labels_to_soft(labels, K)

                    # Rebuild MLP with new output size
                    net = self._build_mlp(D, K).to(device)
                    optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

                    if self.verbose:
                        print(f"[DeepDPM] Epoch {epoch}: {action} -> K = {K}")

            # Update targets periodically (self-training)
            if epoch % 10 == 0 and epoch > 0:
                targets = self._sharpen_targets(probs)

        # Final labels
        with torch.no_grad():
            probs_final = torch.softmax(net(X_t), dim=1).cpu().numpy()
            labels = probs_final.argmax(axis=1)

        # Remap labels to contiguous 0..K-1
        labels, centers, K = self._remap_labels(X, labels)

        return labels, centers, K

    def _build_mlp(self, input_dim: int, n_clusters: int):
        """Build a simple MLP for soft cluster assignment."""
        import torch.nn as nn

        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_clusters))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Split / Merge via Bayesian model selection (Hastings ratio)
    # ------------------------------------------------------------------

    def _propose_split_merge(
        self, X: np.ndarray, labels: np.ndarray, K: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """Evaluate split and merge proposals.

        Uses the Normal-Inverse-Wishart (NIW) marginal log-likelihood
        to compute Hastings ratios for split/merge acceptance.

        Split criterion (DeepDPM Eq. 7):
            H_split = log(alpha) + lgamma(N_1) + ll_1 + lgamma(N_2) + ll_2
                    - lgamma(N_k) - ll_k

        Merge criterion (reverse):
            H_merge = lgamma(N_merged) + ll_merged - log(alpha)
                    - lgamma(N_a) - ll_a - lgamma(N_b) - ll_b
        """
        from scipy.special import gammaln
        from scipy.spatial.distance import cdist

        actions = []

        # --- Try splits ---
        centers = self._compute_centers(X, labels)
        for k in range(K):
            mask = labels == k
            n_k = mask.sum()
            if n_k < 2 * self.min_cluster_size:
                continue

            X_k = X[mask]
            # Split into 2 via KMeans
            from sklearn.cluster import KMeans as _KM
            km2 = _KM(n_clusters=2, random_state=self.random_state, n_init=5)
            sub_labels = km2.fit_predict(X_k)

            n1 = (sub_labels == 0).sum()
            n2 = (sub_labels == 1).sum()
            if n1 < self.min_cluster_size or n2 < self.min_cluster_size:
                continue

            ll_parent = self._niw_log_marginal(X_k)
            ll_child1 = self._niw_log_marginal(X_k[sub_labels == 0])
            ll_child2 = self._niw_log_marginal(X_k[sub_labels == 1])

            H = (np.log(self.alpha) + gammaln(n1) + ll_child1
                 + gammaln(n2) + ll_child2 - gammaln(n_k) - ll_parent)

            if H > 0 or rng.random() < np.exp(min(H, 0)):
                actions.append(("split", k, sub_labels, H))

        # --- Try merges ---
        if K > 2:
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)

            for k_a in range(K):
                k_b = dists[k_a].argmin()
                if k_b <= k_a:
                    continue  # avoid duplicate pairs

                mask_a = labels == k_a
                mask_b = labels == k_b
                n_a, n_b = mask_a.sum(), mask_b.sum()
                if n_a == 0 or n_b == 0:
                    continue

                X_merged = X[mask_a | mask_b]
                ll_a = self._niw_log_marginal(X[mask_a])
                ll_b = self._niw_log_marginal(X[mask_b])
                ll_merged = self._niw_log_marginal(X_merged)

                H = (gammaln(n_a + n_b) + ll_merged - np.log(self.alpha)
                     - gammaln(n_a) - ll_a - gammaln(n_b) - ll_b)

                if H > 0 or rng.random() < np.exp(min(H, 0)):
                    actions.append(("merge", (k_a, k_b), None, H))

        if not actions:
            return labels, centers, K, "none"

        # Pick the action with highest Hastings ratio
        best = max(actions, key=lambda a: a[3])

        if best[0] == "split":
            _, k_split, sub_labels, _ = best
            new_labels = labels.copy()
            mask = labels == k_split
            # Points in subcluster 1 get the new label K
            new_labels[mask] = np.where(sub_labels == 0, k_split, K)
            new_labels, new_centers, new_K = self._remap_labels(X, new_labels)
            return new_labels, new_centers, new_K, f"split(C{k_split})"

        else:  # merge
            _, (k_a, k_b), _, _ = best
            new_labels = labels.copy()
            new_labels[labels == k_b] = k_a
            new_labels, new_centers, new_K = self._remap_labels(X, new_labels)
            return new_labels, new_centers, new_K, f"merge(C{k_a}+C{k_b})"

    def _niw_log_marginal(self, X: np.ndarray) -> float:
        """Compute the Normal-Inverse-Wishart log marginal likelihood.

        This is the key quantity for Bayesian model selection in DeepDPM.
        For a cluster with data X of shape (n, d), computes:

            log p(X | NIW prior) = log Gamma_d(...) - log Gamma_d(...)
                                   + (kappa_0/2) log|Lambda_0|
                                   - (kappa_n/2) log|Lambda_n|
                                   + correction terms

        Uses a weakly informative prior: kappa_0 = 0.01, nu_0 = d + 2,
        Lambda_0 = I * sigma_scale.
        """
        from scipy.special import gammaln, multigammaln

        n, d = X.shape
        if n < 2:
            return 0.0

        # Prior hyperparameters (weakly informative)
        kappa_0 = 0.01
        nu_0 = d + 2
        mu_0 = X.mean(axis=0)
        sigma_scale = 0.01 * np.var(X)
        Lambda_0 = np.eye(d) * max(sigma_scale, 1e-6)

        # Posterior hyperparameters
        x_bar = X.mean(axis=0)
        S = np.cov(X.T, bias=True) * n  # scatter matrix

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        diff = x_bar - mu_0
        Lambda_n = (Lambda_0 + S
                    + (kappa_0 * n / kappa_n) * np.outer(diff, diff))

        # Log marginal likelihood
        try:
            _, logdet_0 = np.linalg.slogdet(Lambda_0)
            _, logdet_n = np.linalg.slogdet(Lambda_n)
        except np.linalg.LinAlgError:
            return 0.0

        log_ml = (
            - (n * d / 2) * np.log(np.pi)
            + multigammaln(nu_n / 2, d) - multigammaln(nu_0 / 2, d)
            + (nu_0 / 2) * logdet_0 - (nu_n / 2) * logdet_n
            + (d / 2) * (np.log(kappa_0) - np.log(kappa_n))
        )

        return float(log_ml)

    # ------------------------------------------------------------------
    # Fallback: parametric split/merge without MLP
    # ------------------------------------------------------------------

    def _parametric_split_merge(
        self, X: np.ndarray, labels: np.ndarray,
        centers: np.ndarray, K: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run split/merge iterations without MLP (GMM-only mode)."""
        n_rounds = max(5, self.max_epochs // self.split_merge_every)
        no_change_count = 0

        for i in range(n_rounds):
            labels_new, centers_new, K_new, action = self._propose_split_merge(
                X, labels, K, rng
            )
            if K_new != K:
                self.split_merge_history_.append({
                    "round": i, "action": action,
                    "k_before": K, "k_after": K_new,
                })
                K = K_new
                labels = labels_new
                centers = centers_new
                no_change_count = 0
                if self.verbose:
                    print(f"[DeepDPM] Round {i}: {action} -> K = {K}")
            else:
                no_change_count += 1
                if no_change_count >= 3:
                    break  # Converged

        return labels, centers, K

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centers(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids."""
        unique = np.unique(labels)
        centers = np.stack([X[labels == k].mean(axis=0) for k in unique])
        return centers

    @staticmethod
    def _labels_to_soft(labels: np.ndarray, K: int) -> np.ndarray:
        """Convert hard labels to label-smoothed soft targets."""
        N = len(labels)
        smooth = 0.1 / K
        targets = np.full((N, K), smooth, dtype=np.float64)
        for i, l in enumerate(labels):
            if l < K:
                targets[i, l] = 1.0 - 0.1 + smooth
        return targets / targets.sum(axis=1, keepdims=True)

    @staticmethod
    def _sharpen_targets(probs: np.ndarray, temperature: float = 0.5) -> np.ndarray:
        """Sharpen soft targets (self-training)."""
        sharpened = probs ** (1.0 / temperature)
        return sharpened / sharpened.sum(axis=1, keepdims=True)

    @staticmethod
    def _remap_labels(
        X: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Remap labels to contiguous 0..K-1 and compute centers."""
        unique = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique)}
        new_labels = np.array([mapping[l] for l in labels])
        K = len(unique)
        centers = np.stack([X[new_labels == k].mean(axis=0) for k in range(K)])
        return new_labels, centers, K

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        k = self.n_clusters_ if self._fitted else "?"
        return (
            f"DeepDPMClustering(init_k={self.init_k}, "
            f"n_clusters={k}, use_mlp={self.use_mlp}, "
            f"status={status})"
        )
