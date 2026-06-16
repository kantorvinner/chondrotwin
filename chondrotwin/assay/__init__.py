from __future__ import annotations

from .decompose import UncertaintySplit, decompose
from .dirichlet import (
    belief,
    expected_probability,
    kl_to_uniform,
    strength,
    vacuity,
)
from .dirichlet_head import DirichletHead, HeadConfig
from .edl import EvidenceConfig, anneal_coefficient, bayes_risk, evidential_loss

__all__ = [
    "DirichletHead",
    "EvidenceConfig",
    "HeadConfig",
    "UncertaintySplit",
    "anneal_coefficient",
    "bayes_risk",
    "belief",
    "decompose",
    "evidential_loss",
    "expected_probability",
    "kl_to_uniform",
    "strength",
    "vacuity",
]
