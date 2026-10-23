"""The H2 ablation figure draws the computed verdict and survives partial artifacts."""

import matplotlib.axes
import pandas as pd
import pytest

import make_headline_figures as figures

ROWS = ("cnn_only", "seg_only", "concat_naive", "fusion_block_pca", "weighted_08_02")


def _write_ablation(folder, cnn_nmi, fusion_nmi, fusion_ci, rows=ROWS):
    """Write an ablation_results.csv with run_ablation's columns into ``folder``."""
    delta = round(fusion_nmi - cnn_nmi, 4)
    values = {"cnn_only": (cnn_nmi, cnn_nmi - 0.005, cnn_nmi + 0.005, 50, 0.0),
              "seg_only": (0.40, 0.39, 0.41, 48, round(0.40 - cnn_nmi, 4)),
              "concat_naive": (cnn_nmi + 0.01, cnn_nmi + 0.005, cnn_nmi + 0.015, 1008, 0.01),
              "fusion_block_pca": (fusion_nmi, fusion_ci[0], fusion_ci[1], 864, delta),
              "weighted_08_02": (fusion_nmi - 0.01, fusion_nmi - 0.015, fusion_nmi - 0.005, 854,
                                 round(delta - 0.01, 4))}
    pd.DataFrame([{"feature_set": r, "nmi": values[r][0], "nmi_ci_low": values[r][1],
                   "nmi_ci_high": values[r][2], "ari": 0.5, "silhouette": 0.1,
                   "n_features": values[r][3], "delta_nmi_vs_cnn": values[r][4]} for r in rows]
                 ).to_csv(folder / "ablation_results.csv", index=False)


@pytest.fixture
def annotations(monkeypatch):
    """Collect the texts passed to Axes.annotate while the figure is drawn."""
    texts = []
    original = matplotlib.axes.Axes.annotate

    def spy(self, text, *args, **kwargs):
        texts.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", spy)
    return texts


def test_not_met_comes_from_the_table(tmp_path, annotations):
    _write_ablation(tmp_path, 0.8765, 0.8776, (0.8643, 0.8963))
    figures.defect_figures(base=str(tmp_path), data_label="NEU-DET")
    assert (tmp_path / "plots" / "h2_ablation.png").exists()
    assert any(t.endswith("H2 not met") for t in annotations)


def test_met_comes_from_the_table(tmp_path, annotations):
    _write_ablation(tmp_path, 0.6947, 0.7985, (0.7959, 0.8034))
    figures.defect_figures(base=str(tmp_path))
    assert any(t.endswith("H2 met") for t in annotations)


def test_a_missing_ci_is_shown_as_not_available(tmp_path, annotations):
    _write_ablation(tmp_path, 0.6947, 0.7985, (float("nan"), float("nan")))
    figures.defect_figures(base=str(tmp_path))
    assert any(t.endswith("H2 n/a") for t in annotations)


def test_a_cnn_only_table_skips_the_figure_without_failing(tmp_path, capsys):
    _write_ablation(tmp_path, 0.6947, 0.7985, (0.7959, 0.8034), rows=("cnn_only",))
    figures.defect_figures(base=str(tmp_path))
    assert not (tmp_path / "plots" / "h2_ablation.png").exists()
    assert "skip h2_ablation" in capsys.readouterr().out
