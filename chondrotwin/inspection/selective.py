from __future__ import annotations

from typing import NamedTuple

import torch


class RiskCoverage(NamedTuple):
    coverage: torch.Tensor
    risk: torch.Tensor


def risk_coverage_curve(uncertainty: torch.Tensor, correct: torch.Tensor) -> RiskCoverage:
    order = torch.argsort(uncertainty)
    ranked_correct = correct[order].float()
    n = ranked_correct.numel()
    cumulative_error = torch.cumsum(1.0 - ranked_correct, dim=0)
    counts = torch.arange(1, n + 1, dtype=torch.float)
    coverage = counts / n
    risk = cumulative_error / counts
    return RiskCoverage(coverage=coverage, risk=risk)


def selective_risk_at_coverage(
    uncertainty: torch.Tensor, correct: torch.Tensor, coverage: float
) -> float:
    curve = risk_coverage_curve(uncertainty, correct)
    index = max(0, min(curve.coverage.numel() - 1, round(coverage * curve.coverage.numel()) - 1))
    return float(curve.risk[index])


def area_under_risk_coverage(uncertainty: torch.Tensor, correct: torch.Tensor) -> float:
    curve = risk_coverage_curve(uncertainty, correct)
    return float(torch.trapz(curve.risk, curve.coverage))
