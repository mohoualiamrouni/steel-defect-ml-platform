"""
Defect Group Clustering Pipeline
=================================

Data-driven regrouping of ~100 individual surface defects into optimal
clusters, using CNN feature vectors (960-dim) and segmentation-derived
shape/texture features.  Proposes an alternative to the 28 expert-defined
groups currently used by the inspection CNN.

References:
    Ren et al. (2022) — Deep Clustering: A Comprehensive Survey
    Caron et al. (2018) — DeepCluster: Unsupervised Learning of Visual Features
    Monti et al. (2003) — Consensus Clustering
"""

__version__ = "0.1.0"
