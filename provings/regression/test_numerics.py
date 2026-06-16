from __future__ import annotations

import numpy as np
import torch

from chondrotwin import build_model, tiny_config
from chondrotwin.inspection.calibration import expected_calibration_error
from chondrotwin.inspection.significance import holm_bonferroni, paired_bootstrap
from chondrotwin.kiln.reproducibility import set_seed
from chondrotwin.kiln.schedule import ScheduleConfig, lr_at


def test_schedule_warmup_peak_and_floor() -> None:
    cfg = ScheduleConfig(warmup_epochs=5, total_epochs=120, peak_lr=2e-4, floor_lr=1e-6)
    assert lr_at(4.0, cfg) == cfg.peak_lr
    assert abs(lr_at(119.0, cfg) - cfg.floor_lr) < 1e-7
    mid = [lr_at(float(e), cfg) for e in range(5, 120)]
    assert all(mid[i] >= mid[i + 1] - 1e-9 for i in range(len(mid) - 1))


def test_calibration_error_extremes() -> None:
    correct = torch.tensor([[0.9, 0.1], [0.85, 0.15]])
    labels = torch.tensor([0, 0])
    low = expected_calibration_error(correct, labels, bins=10)
    overconfident = torch.tensor([[0.99, 0.01], [0.99, 0.01]])
    wrong = torch.tensor([1, 1])
    high = expected_calibration_error(overconfident, wrong, bins=10)
    assert high > low


def test_bootstrap_zero_difference_and_holm() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    same = paired_bootstrap(base, base.copy(), lambda a: float(a.mean()), iterations=500, seed=1)
    assert abs(same.estimate) < 1e-9
    assert same.low <= 0 <= same.high
    better = paired_bootstrap(base, base + 0.5, lambda a: float(a.mean()), iterations=500, seed=1)
    assert better.estimate > 0
    decisions = holm_bonferroni([0.001, 0.2, 0.04], alpha=0.05)
    assert decisions[0] and not decisions[1]


def test_forward_is_deterministic_under_seed() -> None:
    spec = tiny_config().spec
    volume = torch.randn(1, 1, spec.depth, spec.height, spec.width)
    set_seed(123)
    first = build_model(tiny_config())(volume).seg_logits
    set_seed(123)
    second = build_model(tiny_config())(volume).seg_logits
    assert torch.allclose(first, second, atol=1e-6)
