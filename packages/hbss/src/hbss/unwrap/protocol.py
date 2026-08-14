"""The unwrapping-error proposal seam.

Implementations behind this protocol are selected by configuration. Planned:

    UniformProposal          -- symmetric; the Year 1 baseline and the fallback
    LocalRandomWalkProposal  -- +/- 1 cycle on a single component
    LUEPProposal             -- reads precomputed per-component probability tables

READ THIS BEFORE IMPLEMENTING ANYTHING HERE
-------------------------------------------
``propose`` returns the log proposal ratio ``log q(k | k') - log q(k' | k)``
even though that quantity is identically zero for the symmetric uniform
proposal shipping in Year 1.

This is deliberate. Retrofitting the proposal ratio when the LUEP arrives in
Year 3 is exactly where a silent Metropolis-Hastings bug gets introduced, and
such a bug produces plausible-looking, wrong posteriors that no amount of
eyeballing will catch. Return the ratio from the first implementation, test
that it is zero for symmetric proposals, and the seam stays honest.

See HBSS-plan.md section 3.4 and HBSS-dev-plan.md section 1.5.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

Array: TypeAlias = Any  # jax.Array once JAX lands
PRNGKey: TypeAlias = Any


class UnwrapProposal(Protocol):
    """Proposes moves in the discrete unwrapping-error state k."""

    def propose(self, key: PRNGKey, k: Array, features: Any) -> tuple[Array, Array]:
        """Propose a new discrete state.

        Parameters
        ----------
        key
            PRNG key.
        k
            Current integer unwrapping-error state, one entry per connected
            component.
        features
            Per-component features: coherence statistics, size, adjacency, and
            (for the LUEP) the precomputed candidate probability table.

        Returns
        -------
        k_new
            The proposed state.
        log_ratio
            ``log q(k | k_new) - log q(k_new | k)``. Zero for symmetric
            proposals -- but returned explicitly, always.
        """
        ...
