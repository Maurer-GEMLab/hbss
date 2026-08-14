"""Run provenance.

Every run writes a ``run.json`` recording everything needed to reproduce it.
Build this before the science starts -- retrofitting provenance after eighteen
months of results never quite works. See HBSS-dev-plan.md section 2.5.

Required fields (see :class:`RunLedger`):

    git SHA and dirty flag, fully resolved config, input data manifest hashes,
    every RNG seed, package versions, host and GPU, wall time.

Status: skeleton. Flesh out in Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunLedger:
    """Provenance record for a single run."""

    git_sha: str
    git_dirty: bool
    config: dict[str, Any]
    data_manifest: dict[str, str] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    packages: dict[str, str] = field(default_factory=dict)
    host: str = ""
    accelerator: str = ""
    wall_time_s: float = 0.0

    def write(self, run_dir: Path) -> Path:
        """Serialise to ``run_dir/run.json`` and return the path."""
        raise NotImplementedError("Phase 0")


def collect(config: dict[str, Any]) -> RunLedger:
    """Gather provenance for the current process."""
    raise NotImplementedError("Phase 0")
