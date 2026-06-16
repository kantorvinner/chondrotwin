from __future__ import annotations

import torch

from chondrotwin.assay.decompose import decompose
from chondrotwin.assay.dirichlet import expected_probability, strength, vacuity
from chondrotwin.assay.dirichlet_head import DirichletHead, HeadConfig


def test_expected_probability_on_simplex() -> None:
    alpha = torch.rand(4, 14, 4) * 5 + 1
    prob = expected_probability(alpha)
    assert torch.allclose(prob.sum(dim=-1), torch.ones(4, 14), atol=1e-5)


def test_head_outputs_alpha_above_one() -> None:
    head = DirichletHead(HeadConfig(in_dim=8, hidden_dim=6, num_grades=4))
    alpha = head(torch.randn(3, 14, 8))
    assert bool((alpha >= 1.0).all())


def test_vacuity_bounds() -> None:
    alpha = torch.rand(5, 4) * 3 + 1
    v = vacuity(alpha)
    k = alpha.shape[-1]
    assert bool((v > 0).all())
    assert bool((v <= k).all())


def test_uncertainty_decomposition_nonneg_and_additive() -> None:
    alpha = torch.rand(6, 14, 4) * 4 + 1
    split = decompose(alpha)
    assert bool((split.aleatoric >= -1e-5).all())
    assert bool((split.epistemic >= -1e-5).all())
    assert torch.allclose(split.total, split.aleatoric + split.epistemic, atol=1e-4)


def test_higher_evidence_lowers_vacuity() -> None:
    low = torch.ones(1, 4) * 1.2
    high = torch.ones(1, 4) * 20.0
    assert float(vacuity(high)) < float(vacuity(low))
    assert float(strength(high)) > float(strength(low))
