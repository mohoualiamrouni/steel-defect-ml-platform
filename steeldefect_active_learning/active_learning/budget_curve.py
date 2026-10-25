"""Labeling-budget-savings curve — the headline empirical result for AL.

Simulates the active-learning loop on a labelled embedding set: for each
acquisition *strategy* and each labeling *budget*, select ``budget`` points from
a pool, train a lightweight classifier (logistic regression on the embeddings),
and measure macro-F1 on a held-out test split.  Averaging over several seeds
gives a curve of macro-F1 vs. number of labels per strategy, from which we read
the **budget savings**: how many more labels random sampling needs to reach the
macro-F1 that an AL strategy reaches at a reference budget.

This turns the report's asserted 7:1 annotation saving into a measured
quantity.  Strategies implemented on the embeddings directly (no CNN
forward pass needed at simulation time):

* ``random``    — uniform sampling (the baseline);
* ``entropy``   — uncertainty: predictive entropy of a seed model, top-b;
* ``typiclust`` — typicality (Hacohen 2022): most typical point per micro-cluster
                  (cold-start regime);
* ``coreset``   — diversity: greedy k-center / farthest-point (Sener 2018);
* ``badge``     — gradient-embedding k-means++ (Ash 2020), uncertainty+diversity.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Acquisition strategies (return selected pool indices)
# ---------------------------------------------------------------------------

def _seed_indices(y_pool: np.ndarray, n_classes: int, rng) -> np.ndarray:
    """A small stratified seed set (≥1 per present class) for model-based strategies."""
    seed = []
    for c in np.unique(y_pool):
        idx_c = np.where(y_pool == c)[0]
        seed.append(rng.choice(idx_c, size=min(2, len(idx_c)), replace=False))
    return np.concatenate(seed)


def _fit_logreg(X, y):
    from sklearn.linear_model import LogisticRegression
    # (sklearn ≥1.7 dropped the `multi_class` kwarg; multinomial is the default.)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def _select(strategy: str, X_pool: np.ndarray, y_pool: np.ndarray, budget: int,
            n_classes: int, rng) -> np.ndarray:
    n = len(X_pool)
    budget = min(budget, n)
    if strategy == "random":
        return rng.choice(n, size=budget, replace=False)

    if strategy == "typiclust":
        # Cold-start typicality: cluster into `budget` micro-clusters, pick the
        # most typical (highest local density = lowest mean kNN distance) point
        # of each cluster.
        from sklearn.cluster import KMeans
        k = min(budget, n)
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X_pool)
        labels = km.labels_
        chosen = []
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            if len(members) <= 2:
                chosen.append(int(members[0]))   # singleton cluster: take it
                continue
            Xc = X_pool[members]
            kk = min(5, len(members) - 1)
            # mean distance to kk nearest neighbours within the cluster
            d2 = np.sum((Xc[:, None, :] - Xc[None, :, :]) ** 2, axis=-1)
            nn = np.sort(d2, axis=1)[:, 1:kk + 1].mean(axis=1)
            chosen.append(int(members[int(nn.argmin())]))
        return np.array(chosen[:budget])

    if strategy == "coreset":
        # Greedy k-center / farthest-first (Sener 2018).
        first = int(rng.integers(0, n))
        chosen = [first]
        d = np.sum((X_pool - X_pool[first]) ** 2, axis=1)
        while len(chosen) < budget:
            nxt = int(d.argmax())
            chosen.append(nxt)
            d = np.minimum(d, np.sum((X_pool - X_pool[nxt]) ** 2, axis=1))
        return np.array(chosen)

    # model-based strategies need a seed classifier
    seed = _seed_indices(y_pool, n_classes, rng)
    clf = _fit_logreg(X_pool[seed], y_pool[seed])
    proba = clf.predict_proba(X_pool)

    if strategy == "entropy":
        ent = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
        return np.argsort(ent)[::-1][:budget]

    if strategy == "badge":
        # Gradient embedding g_x = (p - onehot(argmax p)) for the linear head,
        # outer-producted with x; k-means++ seeds on it select diverse,
        # high-magnitude (uncertain) points (Ash 2020).
        yhat = proba.argmax(axis=1)
        oneh = np.zeros_like(proba)
        oneh[np.arange(len(proba)), yhat] = 1.0
        resid = proba - oneh                       # (N, C)
        # magnitude-weighted embedding (compact BADGE proxy)
        g = resid[:, :, None] * X_pool[:, None, :]  # (N, C, D)
        g = g.reshape(len(X_pool), -1)
        return _kmeanspp_seed(g, budget, rng)

    raise ValueError(f"unknown strategy '{strategy}'")


def _kmeanspp_seed(G: np.ndarray, k: int, rng) -> np.ndarray:
    """k-means++ seeding: pick k points with D²-weighted sampling."""
    n = len(G)
    first = int(rng.integers(0, n))
    chosen = [first]
    d2 = np.sum((G - G[first]) ** 2, axis=1)
    while len(chosen) < min(k, n):
        probs = d2 / (d2.sum() + 1e-12)
        nxt = int(rng.choice(n, p=probs))
        chosen.append(nxt)
        d2 = np.minimum(d2, np.sum((G - G[nxt]) ** 2, axis=1))
    return np.array(chosen)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def labeling_budget_curve(
    X: np.ndarray,
    y: np.ndarray,
    strategies: Sequence[str] = ("random", "entropy", "typiclust", "coreset", "badge"),
    budgets: Sequence[int] = (20, 50, 100, 200, 400, 800),
    n_seeds: int = 5,
    test_frac: float = 0.30,
    random_state: int = 42,
):
    """Run the AL simulation; return a long-form DataFrame.

    Columns: ``strategy, budget, seed, macro_f1, rare_class_recall``.
    """
    import pandas as pd
    from sklearn.metrics import f1_score, recall_score
    from sklearn.model_selection import train_test_split

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    rows: List[Dict] = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(random_state + seed)
        X_pool, X_te, y_pool, y_te = train_test_split(
            X, y, test_size=test_frac, random_state=random_state + seed, stratify=y
        )
        for strat in strategies:
            for b in budgets:
                if b > len(X_pool):
                    continue
                idx = _select(strat, X_pool, y_pool, b, n_classes, rng)
                if len(np.unique(y_pool[idx])) < 2:
                    continue  # degenerate selection; skip
                clf = _fit_logreg(X_pool[idx], y_pool[idx])
                pred = clf.predict(X_te)
                rows.append({
                    "strategy": strat,
                    "budget": int(b),
                    "seed": int(seed),
                    "macro_f1": float(f1_score(y_te, pred, average="macro")),
                    "rare_class_recall": float(
                        recall_score(y_te, pred, average="macro", zero_division=0)
                    ),
                })
    return pd.DataFrame(rows)


def budget_savings(df, reference_strategy: str = "typiclust",
                   baseline: str = "random", ref_budget: int = 200) -> Dict:
    """Estimate how many more labels *baseline* needs to match *reference* @ref_budget."""
    g = df.groupby(["strategy", "budget"])["macro_f1"].mean().reset_index()
    ref = g[(g.strategy == reference_strategy) & (g.budget == ref_budget)]
    if ref.empty:
        return {}
    target_f1 = float(ref["macro_f1"].iloc[0])
    base = g[g.strategy == baseline].sort_values("budget")
    matching = base[base["macro_f1"] >= target_f1]
    base_budget = int(matching["budget"].iloc[0]) if not matching.empty else None
    ratio = (base_budget / ref_budget) if base_budget else float("inf")
    return {
        "reference_strategy": reference_strategy,
        "ref_budget": ref_budget,
        "target_macro_f1": round(target_f1, 4),
        "baseline": baseline,
        "baseline_budget_to_match": base_budget,
        "savings_ratio": round(ratio, 2) if base_budget else None,
    }


def _ci95_halfwidth(std, count) -> np.ndarray:
    """Half-width of the 95% Student-t interval of a mean: t(0.975, n-1) · std / sqrt(n).

    Returns 0 where fewer than 2 values exist.  With 5 seeds the quantile is
    2.776, not the normal 1.96, which would understate the band by about 30%.
    """
    from scipy import stats

    std = np.nan_to_num(np.asarray(std, dtype=float))
    n = np.asarray(count, dtype=float)
    q = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)), 0.0)
    return q * std / np.sqrt(np.maximum(n, 1))


def plot_budget_curve(df, output_path: str, savings: Dict = None,
                      data_label: str = None) -> None:
    """macro-F1 vs #labels, one line per strategy with a 95% CI band of the mean over seeds.

    ``data_label`` names the data source in the title (for example
    ``"calibrated-synthetic data"``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for strat, gd in df.groupby("strategy"):
        agg = gd.groupby("budget")["macro_f1"].agg(["mean", "std", "count"])
        m = agg["mean"].values
        hw = _ci95_halfwidth(agg["std"].values, agg["count"].values)
        x = agg.index.values
        ax.plot(x, m, marker="o", label=strat)
        ax.fill_between(x, m - hw, m + hw, alpha=0.15)
    ax.set_xlabel("Number of labels (budget)")
    ax.set_ylabel("Macro-F1 (test set)")
    n_seeds = int(df["seed"].nunique()) if "seed" in df.columns else None
    source = f" ({data_label})" if data_label else ""
    seeds = f" over {n_seeds} seeds" if n_seeds else " over seeds"
    ax.set_title(f"Active-learning budget curve{source}\nmean macro-F1 and Student-t 95% CI{seeds}")
    if savings and savings.get("baseline_budget_to_match"):
        ax.annotate(
            f"{savings['baseline']} needs "
            f"{savings['baseline_budget_to_match']} labels to match "
            f"{savings['reference_strategy']}@{savings['ref_budget']} "
            f"(x{savings['savings_ratio']})",
            xy=(0.5, 0.05), xycoords="axes fraction", ha="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.6),
        )
    ax.legend(title="strategy", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
