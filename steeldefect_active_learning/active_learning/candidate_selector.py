"""
Stratified candidate selector combining uncertainty + diversity.

Selection strategies:

stratified (default)
    1. Group images by cluster (diversity from clustering.py).
    2. Within each cluster sort by uncertainty score (descending).
    3. Pick top-K images per cluster where K = budget // n_clusters.
    4. Distribute any remainder to the clusters with the highest
       average uncertainty.

coreset (Sener & Savarese ICLR 2018)
    Greedy k-center: iteratively select the unlabeled point farthest
    from the current labeled set.  Minimizes the covering radius
    δ(S) = max_{x∉S} min_{s∈S} d(x,s) and bounds generalization gap.
    Requires `embeddings_dict` to be provided.

sensitivity (Axiotis et al. 2024)
    Clustering-based sensitivity sampling:
        s(x_i) = d(x_i, μ_{C(i)})² / Φ_k  +  1 / |C(i)|
    Samples proportional to s(x_i); provably preserves average loss
    up to (1±ε) factor.
    Requires `embeddings_dict` to be provided.

badge (Ash et al. 2020)
    Gradient embedding diversity via k-means++ seeding.
    Uses per-image gradient embeddings (∇_{θ_last} CE) computed by
    compute_badge_embeddings() in uncertainty.py.
    Selects a batch that spans the gradient space — combines uncertainty
    (gradient magnitude) with diversity (k-means++ spacing).
    Requires `embeddings_dict` to contain BADGE gradient embeddings.

typiclust (Hacohen et al. 2022)
    Best strategy for low budgets (< 100 labels).
    Clusters unlabeled pool into K=budget clusters; selects the most
    *typical* (most central) point from each cluster.
    High typicality = high density = representative sample.
    Does NOT require precomputed uncertainty scores.
    Requires `embeddings_dict` to be provided.

cal (Margatina et al. 2021, Contrastive Active Learning)
    Selects points that are close to labeled examples in feature space
    but have divergent predictions — contrastive border cases.
    score(x_p) = max_{x_i in KNN_labeled(x_p)} KL(p(y|x_i) || p(y|x_p))
    Requires `embeddings_dict` and `label_probs_dict` (softmax probs for
    both labeled and unlabeled images).

similar (Kothawade et al. 2021 — APPROXIMATION, not true submodular)
    NOTE: canonical SIMILAR uses submodular facility location with
    monotone-submodular gain f(A) = Σ_v max_{a∈A} sim(v,a), which gives
    a (1-1/e) approximation guarantee under greedy. The current
    implementation does NOT compute a submodular set function — it
    applies per-item inverse-frequency class weights to the uncertainty
    score, a diversity heuristic that is fast but has no guarantee:
        w_c = 1 / (count_c + 1)
        score(x) = Σ_c w_c · p_c(x) · uncertainty(x)
    Useful when classes are heavily imbalanced (this use case). For the
    true submodular variant, see `_similar_submodular_select` (requires
    embeddings).
    Requires `label_probs_dict` and `class_counts`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


SelectionMethod = Literal[
    "stratified", "coreset", "sensitivity", "badge", "typiclust", "cal", "similar",
    "similar_submodular", "batchbald", "batchbald_canonical", "powerbald",
]


def _auto_budget_strategy(
    budget: int,
    n_labeled: int,
    class_counts: Optional[Dict[int, int]] = None,
    n_classes: int = 11,
    target_per_class: Optional[List[int]] = None,
) -> str:
    """Phase-aware AL strategy selection (Hacohen et al. 2022).

    Three-phase model based on the insight that typical examples are best
    for low budgets and atypical (uncertain) examples are best for high
    budgets (phase transition, Hacohen 2022 Section 4.2).

    Phases (thresholds scale with n_classes):
        1. Cold-start (n_labeled < 2 * n_classes): TypiClust
           - Need representative coverage of all classes.
        2. Exploration (n_labeled < 10 * n_classes): BADGE
           - Balance uncertainty (gradient magnitude) + diversity (spacing).
        3. Exploitation (n_labeled >= 10 * n_classes):
           - If imbalanced (ratio > 3): SIMILAR (Kothawade 2021)
           - Else: Stratified (uncertainty-focused per cluster)

    Family-adaptive thresholds (closes report limitation L7):
        When ``target_per_class`` is provided, the global thresholds (2C, 10C)
        are replaced by *per-class* checks: phase 1 ends when every class c
        has reached 2 * target_per_class[c]; phase 2 ends when every class
        has reached 10 * target_per_class[c]. This lets rare defect classes
        (coil_break, weld_defect, lamination — < 10 instances/month) demand
        more annotation per class than the majority (clean, scratch).

    Args:
        budget:           Number of images to label this cycle.
        n_labeled:        Current size of labeled set.
        class_counts:     Dict {class_id: count} for labeled set. Required when
                          ``target_per_class`` is provided.
        n_classes:        Number of classes in the problem (default 11 for the plant taxonomy).
        target_per_class: Optional per-class target counts. If given, must be a
                          list/array of length n_classes. Rare classes can be
                          assigned 3× the target of majority classes.

    Returns:
        Recommended method string.

    References:
        Hacohen et al. (2022) "Active Learning on a Budget" — arXiv:2202.02794.
        Kothawade et al. (2021) "SIMILAR" — arXiv:2107.00717.
        Ash et al. (2020) "BADGE" — arXiv:1906.03671.
    """
    # Family-adaptive path: per-class thresholds.
    if target_per_class is not None and class_counts is not None:
        targets = list(target_per_class)
        if len(targets) != n_classes:
            raise ValueError(
                f"target_per_class length {len(targets)} != n_classes {n_classes}"
            )
        # Phase 1 ends when every class has been seen at least 2x its target.
        cold_done = all(class_counts.get(c, 0) >= 2 * targets[c] for c in range(n_classes))
        if not cold_done:
            return "typiclust"
        # Phase 2 ends when every class has reached 10x its target.
        exploration_done = all(
            class_counts.get(c, 0) >= 10 * targets[c] for c in range(n_classes)
        )
        if not exploration_done:
            return "badge"
        # Phase 3: still imbalance-aware.
        counts = np.array(
            [class_counts.get(c, 0) for c in range(n_classes)], dtype=float
        )
        if counts.max() / (counts.min() + 1e-9) > 3.0:
            return "similar"
        return "stratified"

    # Legacy uniform-threshold path (backward compatible).
    cold_start_threshold = 2 * n_classes
    if n_labeled < cold_start_threshold:
        return "typiclust"
    exploration_threshold = 10 * n_classes
    if n_labeled < exploration_threshold:
        return "badge"
    if class_counts is not None and len(class_counts) > 1:
        counts = np.array(list(class_counts.values()), dtype=float)
        imbalance_ratio = counts.max() / (counts.min() + 1e-9)
        if imbalance_ratio > 3.0:
            return "similar"
    return "stratified"


def select_candidates(
    uncertainty_scores: Dict[str, float],
    cluster_assignments: Dict[str, int],
    labeling_budget: int = 200,
    method: SelectionMethod = "stratified",
    embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    already_labeled: Optional[List[str]] = None,
    random_state: int = 42,
    # CAL / SIMILAR extras
    label_probs_dict: Optional[Dict[str, np.ndarray]] = None,
    class_counts: Optional[Dict[int, int]] = None,
    cal_k_neighbors: int = 10,
) -> List[str]:
    """Select the most informative and diverse candidates for labeling.

    Args:
        uncertainty_scores:  Dict {path: uncertainty_score}. Higher = more uncertain.
        cluster_assignments: Dict {path: cluster_id}.
        labeling_budget:     Total number of images to select.
        method:              One of: stratified | coreset | sensitivity | badge |
                             typiclust | cal | similar.
        embeddings_dict:     Required for coreset / sensitivity / badge / typiclust / cal.
                             Dict {path: embedding_vector (1-D np.ndarray)}.
        already_labeled:     Paths already labeled (used as initial centers for coreset).
        random_state:        Random seed (sensitivity / typiclust only).
        label_probs_dict:    Dict {path: softmax_probs (np.ndarray shape [n_classes])}.
                             Required for 'cal' and 'similar' methods. Must cover both
                             labeled (already_labeled) and unlabeled candidates.
        class_counts:        Dict {class_id: count} of the labeled set. Used by 'similar'.
        cal_k_neighbors:     Number of labeled neighbors to use in CAL scoring.

    Returns:
        List of selected image paths (length ≤ labeling_budget).
    """
    if labeling_budget <= 0:
        return []

    if method == "stratified":
        return _stratified_select(
            uncertainty_scores, cluster_assignments, labeling_budget
        )

    elif method == "coreset":
        if embeddings_dict is None:
            raise ValueError("'coreset' method requires embeddings_dict")
        from steeldefect_active_learning.active_learning.clustering import greedy_k_center
        return greedy_k_center(embeddings_dict, labeling_budget, already_labeled)

    elif method == "sensitivity":
        if embeddings_dict is None:
            raise ValueError("'sensitivity' method requires embeddings_dict")
        from steeldefect_active_learning.active_learning.clustering import sensitivity_sample
        n_clusters = max(2, len(set(cluster_assignments.values())))
        return sensitivity_sample(
            embeddings_dict, n_clusters, labeling_budget, random_state
        )

    elif method == "badge":
        if embeddings_dict is None:
            raise ValueError("'badge' method requires embeddings_dict (BADGE gradient embeddings)")
        return _badge_select(embeddings_dict, labeling_budget, random_state)

    elif method == "typiclust":
        if embeddings_dict is None:
            raise ValueError("'typiclust' method requires embeddings_dict")
        return _typiclust_select(embeddings_dict, labeling_budget, random_state)

    elif method == "cal":
        if embeddings_dict is None:
            raise ValueError("'cal' method requires embeddings_dict")
        if label_probs_dict is None:
            raise ValueError("'cal' method requires label_probs_dict")
        if already_labeled is None or len(already_labeled) == 0:
            # Fall back to uncertainty when no labeled set available yet
            return _top_uncertainty_select(uncertainty_scores, labeling_budget)
        return _cal_select(
            embeddings_dict,
            label_probs_dict,
            already_labeled,
            uncertainty_scores,
            labeling_budget,
            cal_k_neighbors,
        )

    elif method == "similar":
        if label_probs_dict is None:
            raise ValueError("'similar' method requires label_probs_dict")
        return _similar_select(
            uncertainty_scores, label_probs_dict, class_counts, labeling_budget
        )

    elif method == "similar_submodular":
        # Canonical SIMILAR: monotone-submodular facility location with the
        # (1−1/e) greedy approximation guarantee (Kothawade et al. 2021,
        # building on Nemhauser 1978). Closes report limitation L5 — the
        # default "similar" method is a fast class-rarity heuristic without
        # this guarantee.
        if embeddings_dict is None:
            raise ValueError("'similar_submodular' method requires embeddings_dict")
        return _similar_submodular_select(embeddings_dict, labeling_budget)

    elif method == "batchbald":
        # BatchBALD requires pre-computed selection from compute_batchbald_scores()
        # which returns ordered paths. Here we just return top-budget by uncertainty
        # as a fallback — the real BatchBALD selection happens in main_pipeline.py
        # when it calls compute_batchbald_scores() (heuristic) or
        # compute_batchbald_canonical() (Kirsch 2019) directly.
        sorted_paths = sorted(uncertainty_scores.keys(),
                              key=lambda p: uncertainty_scores[p], reverse=True)
        return sorted_paths[:labeling_budget]

    elif method == "powerbald":
        # PowerBALD: temperature-softmax sampling without replacement over
        # the existing uncertainty scores (Kirsch et al. 2021, arXiv:2106.12059).
        # Cheap O(B), no MC passes beyond what was already done to populate
        # uncertainty_scores.
        from steeldefect_active_learning.active_learning.uncertainty_batch import (
            powerbald_select_from_scores,
        )
        return powerbald_select_from_scores(
            uncertainty_scores,
            labeling_budget,
            temperature=1.0,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose: "
            "stratified | coreset | sensitivity | badge | typiclust | cal | similar | "
            "batchbald | powerbald"
        )


# ---------------------------------------------------------------------------
# Private strategy implementations
# ---------------------------------------------------------------------------

def _stratified_select(
    uncertainty_scores: Dict[str, float],
    cluster_assignments: Dict[str, int],
    labeling_budget: int,
) -> List[str]:
    """Original stratified uncertainty + diversity selection."""
    paths = list(uncertainty_scores.keys())
    df = pd.DataFrame({
        "path": paths,
        "uncertainty": [uncertainty_scores[p] for p in paths],
        "cluster": [cluster_assignments.get(p, -1) for p in paths],
    })
    df = df.sort_values("uncertainty", ascending=False)

    clusters = sorted(df["cluster"].unique())
    n_clusters = len(clusters)

    if n_clusters == 0:
        return []

    base_budget = labeling_budget // n_clusters
    remainder = labeling_budget % n_clusters

    cluster_mean_unc = (
        df.groupby("cluster")["uncertainty"].mean()
        .sort_values(ascending=False)
    )
    top_clusters_for_remainder = cluster_mean_unc.index[:remainder].tolist()

    selected: List[str] = []
    for cluster_id in clusters:
        cluster_df = df[df["cluster"] == cluster_id]
        extra = 1 if cluster_id in top_clusters_for_remainder else 0
        quota = base_budget + extra
        selected.extend(cluster_df.head(quota)["path"].tolist())

    return selected


def _top_uncertainty_select(
    uncertainty_scores: Dict[str, float],
    labeling_budget: int,
) -> List[str]:
    """Simple top-k by uncertainty (fallback)."""
    sorted_paths = sorted(
        uncertainty_scores.keys(),
        key=lambda p: uncertainty_scores[p],
        reverse=True,
    )
    return sorted_paths[:labeling_budget]


def _badge_select(
    badge_embeddings: Dict[str, np.ndarray],
    labeling_budget: int,
    random_state: int = 42,
) -> List[str]:
    """BADGE: k-means++ seeding on gradient embeddings (Ash et al. 2020).

    Selects a diverse batch by treating gradient embeddings as points and
    applying k-means++ initialization. Points with large gradient norms
    (uncertain) and far from already-selected points (diverse) are preferred.

    The gradient magnitude encodes uncertainty; k-means++ spacing encodes
    diversity — so BADGE combines both in a single selection pass.

    Args:
        badge_embeddings: Dict {path: gradient_embedding (1-D np.ndarray)}.
        labeling_budget:  Number of points to select.
        random_state:     Random seed.

    Returns:
        List of selected paths.
    """
    rng = np.random.default_rng(random_state)
    paths = list(badge_embeddings.keys())
    if len(paths) <= labeling_budget:
        return paths

    G = np.stack([badge_embeddings[p] for p in paths])  # (N, d)
    N = len(paths)
    budget = min(labeling_budget, N)

    # k-means++ seeding: first center sampled proportional to ||g||²
    norms_sq = np.sum(G ** 2, axis=1)  # (N,)
    probs = norms_sq / (norms_sq.sum() + 1e-12)
    first_idx = rng.choice(N, p=probs)

    selected_indices = [first_idx]
    # D²[i] = min distance² from point i to any selected center
    D_sq = np.sum((G - G[first_idx]) ** 2, axis=1)  # (N,)

    for _ in range(budget - 1):
        D_sq_sum = D_sq.sum()
        if D_sq_sum < 1e-12:
            # All remaining points are identical; fill randomly
            remaining = list(set(range(N)) - set(selected_indices))
            selected_indices.append(int(rng.choice(remaining)))
            break
        sample_probs = D_sq / D_sq_sum
        next_idx = int(rng.choice(N, p=sample_probs))
        selected_indices.append(next_idx)
        # Update D² with distances to new center
        new_dists = np.sum((G - G[next_idx]) ** 2, axis=1)
        D_sq = np.minimum(D_sq, new_dists)

    return [paths[i] for i in selected_indices]


def _typiclust_select(
    embeddings_dict: Dict[str, np.ndarray],
    labeling_budget: int,
    random_state: int = 42,
) -> List[str]:
    """TypiClust: select the most typical (densest) point per cluster.

    Algorithm (Hacohen et al. 2022, NeurIPS):
      1. Cluster the unlabeled pool into K = labeling_budget clusters
         with k-means.
      2. For each cluster, compute typicality via k-NN density:
         typical(x) = 1 / (1 + mean_{x' ∈ kNN(x)} ||x - x'||)
      3. Pick argmax typicality within each cluster.

    Paper equation (Sec. 3.2): typicality uses the k-nearest-neighbour
    density estimator restricted to the same cluster, NOT distance to
    the centroid. Centroid-distance picks the point nearest the mean,
    which is only a density proxy under spherical Gaussian clusters and
    breaks down for elongated or multimodal clusters.

    Args:
        embeddings_dict:  Dict {path: embedding (1-D np.ndarray)}.
        labeling_budget:  K = number of clusters = number of points selected.
        random_state:     Random seed for k-means.

    Returns:
        List of selected paths (one per cluster).
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError("TypiClust requires scikit-learn: pip install scikit-learn") from e

    paths = list(embeddings_dict.keys())
    if len(paths) <= labeling_budget:
        return paths

    X = np.stack([embeddings_dict[p] for p in paths])  # (N, d)
    K = min(labeling_budget, len(paths))

    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    selected = []
    for k in range(K):
        mask = labels == k
        if not mask.any():
            continue
        cluster_paths = [paths[i] for i in range(len(paths)) if mask[i]]
        cluster_X = X[mask]

        # k-NN density within the cluster (Hacohen 2022 eq. 1)
        # k_nn is the neighbourhood size — paper uses 20 or |cluster|/2.
        n_in_cluster = len(cluster_X)
        k_nn = max(1, min(20, n_in_cluster - 1))
        if k_nn < 1:
            # Singleton cluster: typicality is meaningless, pick the sole point
            selected.append(cluster_paths[0])
            continue

        nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(cluster_X)
        # First neighbour is the point itself (distance 0), drop it
        dists, _ = nn.kneighbors(cluster_X)
        mean_knn_dist = dists[:, 1:].mean(axis=1)
        typicality = 1.0 / (1.0 + mean_knn_dist)
        best_local = int(np.argmax(typicality))
        selected.append(cluster_paths[best_local])

    return selected


def _cal_select(
    embeddings_dict: Dict[str, np.ndarray],
    label_probs_dict: Dict[str, np.ndarray],
    labeled_paths: List[str],
    uncertainty_scores: Dict[str, float],
    labeling_budget: int,
    k_neighbors: int = 10,
) -> List[str]:
    """CAL: Contrastive Active Learning (Margatina et al. 2021).

    Selects unlabeled points that are:
      - Close to labeled examples in embedding space (contrastive)
      - But have very different predictions (high KL divergence)

    score(x_p) = max_{x_i in KNN_labeled(x_p, k)} KL(p(y|x_i) || p(y|x_p))

    This targets the decision boundary from the labeled side, finding
    the most confusing examples near already-labeled regions.

    Args:
        embeddings_dict:   Dict {path: embedding} for ALL images.
        label_probs_dict:  Dict {path: softmax_probs} for ALL images.
        labeled_paths:     Paths in the current labeled set.
        uncertainty_scores: Fallback uncertainty scores for unlabeled paths.
        labeling_budget:   Number of images to select.
        k_neighbors:       Number of labeled neighbors to consider (default 10).

    Returns:
        List of selected paths.
    """
    unlabeled = [p for p in embeddings_dict if p not in set(labeled_paths)]
    if len(unlabeled) == 0:
        return []

    # Filter labeled_paths to those with embeddings
    labeled_valid = [p for p in labeled_paths if p in embeddings_dict]
    if len(labeled_valid) == 0:
        return _top_uncertainty_select(uncertainty_scores, labeling_budget)

    k = min(k_neighbors, len(labeled_valid))

    # Build labeled embedding matrix
    L_paths = labeled_valid
    L_emb = np.stack([embeddings_dict[p] for p in L_paths])  # (|L|, d)
    L_emb_norm = L_emb / (np.linalg.norm(L_emb, axis=1, keepdims=True) + 1e-12)

    cal_scores: Dict[str, float] = {}

    for p_u in unlabeled:
        if p_u not in label_probs_dict:
            cal_scores[p_u] = uncertainty_scores.get(p_u, 0.0)
            continue

        e_u = embeddings_dict[p_u]
        e_u_norm = e_u / (np.linalg.norm(e_u) + 1e-12)

        # Cosine distance to each labeled point
        cos_sims = L_emb_norm @ e_u_norm  # (|L|,)
        knn_indices = np.argpartition(-cos_sims, min(k, len(cos_sims) - 1))[:k]

        # KL( p_labeled || p_unlabeled ) — "how much does the neighbor disagree?"
        p_u_probs = label_probs_dict[p_u].astype(np.float64)
        p_u_probs = np.clip(p_u_probs, 1e-12, 1.0)

        max_kl = 0.0
        for idx in knn_indices:
            p_i_probs = label_probs_dict[L_paths[idx]].astype(np.float64)
            p_i_probs = np.clip(p_i_probs, 1e-12, 1.0)
            # KL(p_i || p_u) = Σ p_i * log(p_i / p_u)
            kl = float(np.sum(p_i_probs * (np.log(p_i_probs) - np.log(p_u_probs))))
            if kl > max_kl:
                max_kl = kl

        cal_scores[p_u] = max_kl

    # Select top-budget by CAL score
    sorted_unlabeled = sorted(unlabeled, key=lambda p: cal_scores.get(p, 0.0), reverse=True)
    return sorted_unlabeled[:labeling_budget]


def _similar_select(
    uncertainty_scores: Dict[str, float],
    label_probs_dict: Dict[str, np.ndarray],
    class_counts: Optional[Dict[int, int]],
    labeling_budget: int,
) -> List[str]:
    """SIMILAR: Imbalance-aware uncertainty weighting (Kothawade et al. 2021).

    Computes a class-reweighted uncertainty score that upweights predictions
    on rare classes, helping to fix class imbalance during active learning.

    score(x) = Σ_c w_c · p_c(x) · uncertainty(x)
    where w_c = 1 / (count_c + 1)   (inverse frequency weight)

    When class_counts is None, falls back to plain uncertainty.

    Args:
        uncertainty_scores: Dict {path: uncertainty (e.g. entropy)}.
        label_probs_dict:   Dict {path: softmax_probs (np.ndarray)}.
        class_counts:       Dict {class_id: count} of the labeled set.
                            If None, uniform weights are used.
        labeling_budget:    Number of images to select.

    Returns:
        List of selected paths.
    """
    if not label_probs_dict:
        return _top_uncertainty_select(uncertainty_scores, labeling_budget)

    # Build inverse-frequency class weights
    if class_counts is not None and len(class_counts) > 0:
        n_classes = max(class_counts.keys()) + 1
        class_weights = np.ones(n_classes, dtype=np.float64)
        for c, cnt in class_counts.items():
            if 0 <= c < n_classes:
                class_weights[c] = 1.0 / (cnt + 1.0)
        # Normalize so weights sum to n_classes (preserve scale)
        class_weights = class_weights / class_weights.mean()
    else:
        class_weights = None

    similar_scores: Dict[str, float] = {}
    for path, base_unc in uncertainty_scores.items():
        if path not in label_probs_dict or class_weights is None:
            similar_scores[path] = base_unc
            continue

        probs = label_probs_dict[path].astype(np.float64)
        n_c = len(probs)
        w = class_weights[:n_c] if n_c <= len(class_weights) else class_weights
        if len(w) < n_c:
            w = np.concatenate([w, np.ones(n_c - len(w))])
        # Weighted sum over classes: high weight → rare class contribution amplified
        weighted = float(np.dot(w, probs))
        similar_scores[path] = base_unc * weighted

    sorted_paths = sorted(
        uncertainty_scores.keys(),
        key=lambda p: similar_scores.get(p, 0.0),
        reverse=True,
    )
    return sorted_paths[:labeling_budget]


def _similar_submodular_select(
    embeddings_dict: Dict[str, np.ndarray],
    labeling_budget: int,
) -> List[str]:
    """True submodular facility-location selection (Kothawade et al. 2021).

    Greedy submodular maximisation of the facility-location gain:

        f(A) = Σ_{v ∈ V} max_{a ∈ A} sim(v, a)

    with cosine similarity as sim. Greedy achieves a (1 - 1/e)
    approximation to the optimum thanks to the monotone-submodular
    structure (Nemhauser 1978).

    Args:
        embeddings_dict:  Dict {path: embedding (1-D np.ndarray)}. The pool V.
        labeling_budget:  Number of items to pick (greedy iterations).

    Returns:
        List of selected paths, ordered by selection step.
    """
    paths = list(embeddings_dict.keys())
    if len(paths) <= labeling_budget:
        return paths

    X = np.stack([embeddings_dict[p] for p in paths]).astype(np.float32)
    # Cosine similarity = normalised dot product
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Xn = X / norms
    sim = Xn @ Xn.T  # (N, N) in [-1, 1]

    N = sim.shape[0]
    max_sim_to_A = np.full(N, -np.inf, dtype=np.float32)  # current max_{a∈A} sim(v,a)
    selected: List[int] = []

    for _ in range(min(labeling_budget, N)):
        # Marginal gain of adding candidate i: Σ_v max(max_sim_to_A[v], sim[v,i]) - sum max_sim_to_A
        # Equivalent to: Σ_v max(0, sim[v,i] - max_sim_to_A[v])
        # For the first pick where max_sim_to_A is all -inf, use column sum.
        if not selected:
            gains = sim.sum(axis=0)
        else:
            gains = np.maximum(sim - max_sim_to_A[:, None], 0.0).sum(axis=0)
            gains[selected] = -np.inf  # exclude already-picked
        best = int(np.argmax(gains))
        selected.append(best)
        max_sim_to_A = np.maximum(max_sim_to_A, sim[:, best])

    return [paths[i] for i in selected]


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def candidates_to_dataframe(
    candidates: List[str],
    uncertainty_scores: Dict[str, float],
    cluster_assignments: Dict[str, int],
    sensitivity_scores: Optional[Dict[str, float]] = None,
    selection_method: str = "stratified",
) -> pd.DataFrame:
    """Build a DataFrame of selected candidates for export.

    Columns: path, uncertainty, cluster, rank_in_cluster,
             sensitivity_score, selection_method
    """
    rows = []
    cluster_ranks: Dict[int, int] = defaultdict(int)

    for path in candidates:
        cid = cluster_assignments.get(path, -1)
        cluster_ranks[cid] += 1
        rows.append({
            "path": path,
            "uncertainty": uncertainty_scores.get(path, float("nan")),
            "cluster": cid,
            "rank_in_cluster": cluster_ranks[cid],
            "sensitivity_score": sensitivity_scores.get(path, float("nan"))
                if sensitivity_scores else float("nan"),
            "selection_method": selection_method,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["cluster", "rank_in_cluster"])
    return df
