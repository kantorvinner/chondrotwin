from __future__ import annotations

import pytest
import torch

from chondrotwin.assay.dirichlet import kl_to_uniform
from chondrotwin.inspection.ordinal import quadratic_weighted_kappa
from chondrotwin.kiln.segmentation_loss import soft_dice


def test_kl_matches_torch_distributions() -> None:
    alpha = torch.rand(8, 4) * 6 + 1
    dist = torch.distributions.Dirichlet(alpha)
    uniform = torch.distributions.Dirichlet(torch.ones_like(alpha))
    reference = torch.distributions.kl_divergence(dist, uniform)
    assert torch.allclose(kl_to_uniform(alpha), reference, atol=1e-4)


def test_kappa_matches_sklearn() -> None:
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    torch.manual_seed(1)
    grade = torch.randint(0, 4, (200,))
    pred = grade.clone()
    flip = torch.rand(200) < 0.3
    pred[flip] = torch.randint(0, 4, (int(flip.sum()),))
    onehot = torch.zeros(200, 4)
    onehot.scatter_(1, pred.unsqueeze(1), 5.0)
    alpha = onehot + 1.0
    mine = quadratic_weighted_kappa(alpha, grade, 4)
    reference = sklearn_metrics.cohen_kappa_score(grade.numpy(), pred.numpy(), weights="quadratic")
    assert abs(mine - reference) < 1e-6


def test_soft_dice_matches_monai() -> None:
    monai_losses = pytest.importorskip("monai.losses")
    torch.manual_seed(2)
    logits = torch.randn(2, 7, 4, 8, 8)
    target = torch.randint(0, 7, (2, 4, 8, 8))
    reference = monai_losses.DiceLoss(
        softmax=True, to_onehot_y=True, smooth_nr=1e-5, smooth_dr=1e-5
    )
    ref_value = float(reference(logits, target.unsqueeze(1)))
    assert abs(float(soft_dice(logits, target)) - ref_value) < 5e-2
