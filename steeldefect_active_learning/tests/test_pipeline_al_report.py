"""PipelineActiveLearner renders its PDF report and its beta-history figure."""

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from steeldefect_active_learning.active_learning.pipeline_al import PipelineActiveLearner


class _Images(Dataset):
    """Tiny (tensor, path) dataset: the batch format compute_uncertainty expects."""

    def __init__(self, n=12):
        self.x = torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        self.paths = [f"img_{i:02d}.png" for i in range(n)]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.x[i], self.paths[i]


def _learner(n_stages):
    torch.manual_seed(0)
    models = [nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3 + d)) for d in range(n_stages)]
    return PipelineActiveLearner(models=models)


def _run_cycles(learner, n_cycles):
    loader = DataLoader(_Images(), batch_size=4)
    for c in range(n_cycles):
        learner.select_candidates(loader, budget=2)
        with torch.no_grad():  # stand-in for retraining: rescale logits so certainty moves
            for model in learner.models:
                model[1].weight.mul_(0.5 + 0.2 * c)
        learner.step(loader)


@pytest.mark.parametrize("n_stages, n_cycles", [(1, 0), (2, 3)])
def test_generate_report_writes_a_pdf(tmp_path, n_stages, n_cycles):
    learner = _learner(n_stages)
    _run_cycles(learner, n_cycles)
    out = tmp_path / "pipeline_al_report.pdf"
    learner.generate_report(str(out))
    assert out.stat().st_size > 0


def test_beta_history_plot_renders(tmp_path):
    learner = _learner(2)
    _run_cycles(learner, 3)
    out = tmp_path / "beta_history.png"
    learner.plot_beta_history(output_path=str(out))
    assert out.stat().st_size > 0
