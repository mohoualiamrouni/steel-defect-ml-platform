"""Validate P1 (active learning) and P2 (clustering + CNN/morphology fusion) on the
REAL public NEU-DET dataset, reusing the existing pipeline functions on precomputed
real features. Honest provenance: source_version='neu-det-v1' (public, NOT plant data).

  CNN embedding : torchvision EfficientNet-B0 (ImageNet), GAP + L2  -> mirrors the inspection CNN
  Morphology    : 24/48 regionprops+Hu+GLCM (defect_clustering.features) on Otsu masks
  P2 (H2)       : defect_clustering.analysis.group_analysis.run_ablation (cnn vs fusion)
  P1 (H1)       : steeldefect_active_learning.active_learning.budget_curve.* (savings curve)

Outputs (isolated, synthetic untouched):
  defect_clustering/outputs_neu/{ablation_results.csv, cluster_metrics.json}
  steeldefect_active_learning/outputs_neu/{al_budget_savings.json, al_budget_curve.csv}
  outputs_neu_p1p2_summary.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SRC = "neu-det-v1"
NEU_ROOT = "steeldefect_ssl/data/neu_det/raw"
DC_OUT = "defect_clustering/outputs_neu"
AL_OUT = "steeldefect_active_learning/outputs_neu"
os.makedirs(DC_OUT, exist_ok=True); os.makedirs(AL_OUT, exist_ok=True)


def list_data():
    classes = sorted(d for d in os.listdir(NEU_ROOT) if os.path.isdir(os.path.join(NEU_ROOT, d)))
    paths, y = [], []
    for i, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(NEU_ROOT, c))):
            if f.lower().endswith((".jpg", ".png", ".bmp", ".jpeg")):
                paths.append(os.path.join(NEU_ROOT, c, f)); y.append(i)
    return classes, paths, np.array(y, dtype=int)


def cnn_embeddings(paths, cache=os.path.join(DC_OUT, "neu_cnn.npy")):
    if os.path.exists(cache):
        return np.load(cache)
    import torch, torchvision
    from torch import nn
    from PIL import Image
    w = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=w); model.classifier = nn.Identity()
    model.eval(); tf = w.transforms()
    feats = []
    with torch.no_grad():
        batch = []
        for i, p in enumerate(paths):
            batch.append(tf(Image.open(p).convert("RGB")))
            if len(batch) == 64 or i == len(paths) - 1:
                x = torch.stack(batch)
                e = model(x).numpy()
                feats.append(e); batch = []
                print(f"  CNN embed {i + 1}/{len(paths)}", end="\r")
    X = np.concatenate(feats).astype("float32")
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)   # L2
    np.save(cache, X); print(f"\n  CNN embeddings {X.shape} -> {cache}")
    return X


def morphology(paths, cache=os.path.join(DC_OUT, "neu_morph.npy")):
    if os.path.exists(cache):
        return np.load(cache)
    import cv2
    from defect_clustering.features.segmentation_features import extract_batch
    masks, intens = [], []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        _, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if m.mean() > 127:           # ensure defect is foreground (minority)
            m = 255 - m
        masks.append(m); intens.append(img)
    X = np.nan_to_num(extract_batch(masks, intensity_images=intens,
                                    aggregation="mean_max", n_jobs=-1)).astype("float32")
    np.save(cache, X); print(f"  morphology {X.shape} -> {cache}")
    return X


def main():
    classes, paths, y = list_data()
    print(f"NEU-DET: {len(paths)} images, {len(classes)} classes {classes}")
    X_cnn = cnn_embeddings(paths)
    X_morph = morphology(paths)
    K = len(classes)

    # ---- P2: clustering + CNN/morphology fusion ablation (H2) ----
    from defect_clustering.analysis.group_analysis import run_ablation
    print("\n[P2] fusion ablation (CNN vs morphology vs fusion) ...")
    abl = run_ablation(cnn_features=X_cnn, seg_features=X_morph, labels_expert=y,
                       k=K, random_state=42, n_bootstrap=100)
    abl.to_csv(os.path.join(DC_OUT, "ablation_results.csv"), index=False)
    print(abl.to_string(index=False))
    row = {r["feature_set"]: r for _, r in abl.iterrows()}
    cnn_nmi = float(row["cnn_only"]["nmi"]); fus = row["fusion_block_pca"]
    h2 = bool(fus["nmi_ci_low"] > cnn_nmi and fus["delta_nmi_vs_cnn"] >= 0.03)
    cluster_metrics = {"source_version": SRC, "n_samples": len(y), "n_clusters": K,
                       "cnn_only_nmi": cnn_nmi, "fusion_nmi": float(fus["nmi"]),
                       "fusion_nmi_ci": [float(fus["nmi_ci_low"]), float(fus["nmi_ci_high"])],
                       "delta_nmi": float(fus["delta_nmi_vs_cnn"]),
                       "fusion_ari": float(fus["ari"]), "H2_pass": h2}
    json.dump(cluster_metrics, open(os.path.join(DC_OUT, "cluster_metrics.json"), "w"), indent=2)

    # ---- P1: active-learning budget-vs-F1 savings (H1) ----
    from steeldefect_active_learning.active_learning.budget_curve import (
        labeling_budget_curve, budget_savings)
    print("\n[P1] active-learning budget curve ...")
    df = labeling_budget_curve(X_cnn, y, budgets=(20, 50, 100, 200, 400, 800),
                               n_seeds=5, random_state=42)
    df.to_csv(os.path.join(AL_OUT, "al_budget_curve.csv"), index=False)
    sav = budget_savings(df, reference_strategy="typiclust", baseline="random", ref_budget=100)
    sav["source_version"] = SRC
    json.dump(sav, open(os.path.join(AL_OUT, "al_budget_savings.json"), "w"), indent=2)
    print("  savings:", json.dumps(sav))

    summary = {"source_version": SRC, "dataset": "NEU-DET (public)", "classes": classes,
               "P2_H2": cluster_metrics, "P1_H1": sav}
    json.dump(summary, open("outputs_neu_p1p2_summary.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"  P2/H2: CNN-only NMI {cnn_nmi:.3f} -> fusion {fus['nmi']:.3f} "
          f"(Δ{fus['delta_nmi_vs_cnn']:+.3f}), H2_pass={h2}")
    print(f"  P1/H1: typiclust@{sav['ref_budget']} matches random@{sav.get('baseline_budget_to_match')} "
          f"-> savings x{sav.get('savings_ratio')}")


if __name__ == "__main__":
    main()
