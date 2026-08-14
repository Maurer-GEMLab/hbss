"""Placeholder so the suite is non-empty from commit one.

The tests that matter here are the overfit-one-batch smoke test, the leakage
guard on train/val/test splitting, and the calibration gates on artifact
export. See HBSS-dev-plan.md section 2.4.
"""

from __future__ import annotations

import hbss_learned


def test_package_imports() -> None:
    assert hbss_learned.__version__
