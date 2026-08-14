"""Placeholder so the suite is non-empty from commit one.

Delete once there are real tests. The tests that matter -- the dense-solve
reference for the smoother and simulation-based calibration -- arrive in
Phase 1; see HBSS-dev-plan.md section 2.2.
"""

from __future__ import annotations

import hbss


def test_package_imports() -> None:
    assert hbss.__version__
