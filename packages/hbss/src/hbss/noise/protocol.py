"""The observation-noise seam.

Implementations behind this protocol are selected by configuration, so swapping
a parametric model for the learned one is a config diff rather than a code
change. Planned implementations:

    ConstantNoise    -- a single scalar variance; the trivial baseline
    CoherenceNoise   -- the standard R proportional to 1/gamma^2 parameterisation
    LearnedNoise     -- reads LNCM-predicted variance fields from the tile store

Note that ``LearnedNoise`` reads *precomputed* arrays. The LNCM is applied
offline and its output written into Zarr alongside the observations, so this
package never imports torch. See HBSS-dev-plan.md section 1.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

Array: TypeAlias = Any  # jax.Array once JAX lands; kept loose in the shell repo


@dataclass(frozen=True)
class Diagonal:
    """Per-observation variance, shape broadcastable to the observation vector."""

    variance: Array


@dataclass(frozen=True)
class Dense:
    """Full observation covariance matrix."""

    covariance: Array


ObsCovariance: TypeAlias = Diagonal | Dense
"""Covariance representation.

Deliberately a sum type from day one even though only ``Diagonal`` is planned
for v0.1. Adding off-diagonal support later is then an additive change rather
than a breaking one -- see HBSS-plan.md section 6, decision 2.
"""


class NoiseModel(Protocol):
    """Maps observations and their covariates to an observation covariance."""

    def covariance(self, obs: Any, covariates: Any) -> ObsCovariance:
        """Return R for this batch of observations.

        Parameters
        ----------
        obs
            The observation batch (sensor, epoch, pixel selection).
        covariates
            Per-pixel and per-acquisition covariates: coherence, topography,
            perpendicular baseline, NWP troposphere, land cover, time of year.
        """
        ...
