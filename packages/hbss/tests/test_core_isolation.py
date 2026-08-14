"""The dependency rule, as an executable test.

``hbss`` must never import torch, directly or transitively. This is what lets
the learned components be deleted without touching the baseline, and it is what
makes the "Year 1 completes without ML" fallback real rather than aspirational.

This test catches the mistake locally and quickly. The authoritative check is
the ``core-isolation`` CI job, which installs ``hbss`` alone into an
environment where torch is not present at all -- there, an accidental import is
an ImportError rather than something a test has to notice.

See HBSS-dev-plan.md section 1.3.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import hbss

FORBIDDEN = ("torch", "torchvision", "hbss_learned")


def test_importing_hbss_does_not_pull_in_torch() -> None:
    for name in FORBIDDEN:
        assert name not in sys.modules, (
            f"importing hbss pulled in {name!r}; "
            "the core package must stay free of the learned stack"
        )


def test_no_submodule_imports_torch() -> None:
    """Import every submodule, then re-check."""
    for info in pkgutil.walk_packages(hbss.__path__, prefix="hbss."):
        importlib.import_module(info.name)

    offenders = [name for name in FORBIDDEN if name in sys.modules]
    assert not offenders, (
        f"an hbss submodule imports {offenders}; "
        "learned models must be consumed as exported artifacts, not imports "
        "(see HBSS-dev-plan.md section 1.6)"
    )
