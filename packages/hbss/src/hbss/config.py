"""Typed run configuration.

The configuration selects which implementation sits behind each seam -- which
noise model, which unwrapping proposal, which observation operators. That is
what makes an experiment a config diff rather than a branch or a fork, and it
is the mechanism the ``experiments/`` sandbox relies on.

Every run resolves its config fully and stores the result in the run ledger, so
that a result can always be traced back to the exact settings that produced it.

Status: skeleton. Flesh out in Phase 0 alongside :mod:`hbss.ledger`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NoiseConfig:
    """Selects and parameterises a :class:`hbss.noise.protocol.NoiseModel`."""

    kind: str = "coherence"  # constant | coherence | learned
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposalConfig:
    """Selects an :class:`hbss.unwrap.protocol.UnwrapProposal`."""

    kind: str = "uniform"  # uniform | local_random_walk | luep
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level configuration for one inference run."""

    noise: NoiseConfig = field(default_factory=NoiseConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        raise NotImplementedError("Phase 0")
