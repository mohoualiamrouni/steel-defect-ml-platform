"""Generate UMAP plots colored by (a) automatic cluster, (b) expert group."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "outputs" / "feature_matrix.npz"
ASSIGN = ROOT / "outputs" / "cluster_assignments.csv"
OUT = ROOT.parents[0] / "tn-report" / "template_tex" / "figures"


def main() -> None:
    data = np.load(FEAT, allow_pickle=True)
    X = data["X"] if "X" in data.files else data[data.files[0]]
    df = pd.read_csv(ASSIGN)

    if X.shape[0] != len(df):
        n = min(X.shape[0], len(df))
        X = X[:n]
        df = df.iloc[:n].reset_index(drop=True)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    Z = reducer.fit_transform(X)

    for col, fname, title in [
        ("cluster_id", "dc_umap_proposed.png", "UMAP — clusters automatiques"),
        ("expert_group", "dc_umap_expert.png", "UMAP — groupes experts"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = df[col].astype("category")
        ax.scatter(
            Z[:, 0], Z[:, 1], c=labels.cat.codes, cmap="tab20",
            s=5, alpha=0.6, linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {OUT / fname}")


if __name__ == "__main__":
    main()
