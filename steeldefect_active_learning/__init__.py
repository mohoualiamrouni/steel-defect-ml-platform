"""
SteelDefect Active Learning + A/B Testing Pipeline
=============================================
Surface defect inspection pipeline for the line inspection system.

Modules:
    config          - Pipeline configuration (YAML-backed dataclass)
    data            - Image loading and PyTorch Dataset
    models          - Model loading (ResNet / DenseNet / U-Net)
    active_learning - Uncertainty scoring, feature extraction, clustering, candidate selection
    ab_testing      - Two-model comparison pipeline and metrics
    pipeline        - Full end-to-end orchestration
"""

__version__ = "0.1.0"
