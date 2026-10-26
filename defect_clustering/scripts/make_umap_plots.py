"""Generate UMAP plots colored by (a) automatic cluster, (b) expert group.

Reads the training outputs in defect_clustering/outputs/ and writes the two PNGs
to defect_clustering/outputs/figures/.
"""
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
OUT = ROOT / "outputs" / "figures"


def pair_rows(n_rows: int, image_ids, df: pd.DataFrame) -> pd.DataFrame:
    """Return the assignment rows of the n_rows feature rows, in their order.

    A default training run clusters a stratified subsample (subsample_n, 8000)
    but assigns every sample, so row i of the two files holds different images.
    Training saves the image_id of each feature row to pair them; older files
    without it can only be paired by position, when their lengths match.
    """
    if image_ids is None:
        if n_rows != len(df):
            raise ValueError(f"{FEAT.name} has {n_rows} rows and no image_ids, but {ASSIGN.name} "
                             f"has {len(df)} rows; rerun the training pipeline, which saves the "
                             "image_ids with the features")
        return df
    ids = pd.Index(np.asarray(image_ids).astype(str))
    if len(ids) != n_rows:
        raise ValueError(f"{FEAT.name} has {n_rows} feature rows but {len(ids)} image_ids")
    by_id = df.set_index(df["image_id"].astype(str))
    if not by_id.index.is_unique or not ids.isin(by_id.index).all():
        raise ValueError(f"the image_ids of {FEAT.name} do not each match one row of "
                         f"{ASSIGN.name}; regenerate both from the same training run")
    return by_id.loc[ids].reset_index(drop=True)


def main() -> None:
    data = np.load(FEAT, allow_pickle=True)
    X = data["X"] if "X" in data.files else data["features"]
    image_ids = data["image_ids"] if "image_ids" in data.files else None
    df = pair_rows(X.shape[0], image_ids, pd.read_csv(ASSIGN))
    OUT.mkdir(parents=True, exist_ok=True)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    Z = reducer.fit_transform(X)

    for col, fname, title in [
        ("cluster_id", "dc_umap_proposed.png", "UMAP: automatic clusters"),
        ("expert_group", "dc_umap_expert.png", "UMAP: expert groups"),
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
