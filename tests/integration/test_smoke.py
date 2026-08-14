"""Cross-package integration tests (tier 2, nightly).

What lands here, per HBSS-dev-plan.md section 2.3, on 10x10-pixel by 10-epoch
synthetic scenes:

  * 95% credible intervals cover the truth for 90-98% of pixels
  * R-hat < 1.01 and ESS above a floor on all hyperparameters
  * velocity RMSE below a fixed threshold
  * the identifiability control -- a scene with elevation-correlated
    *deformation* and zero troposphere must not be absorbed into the
    tropospheric state

That last one is a test, not a figure someone remembers to make.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_placeholder() -> None:
    pytest.skip("integration tests arrive in Phase 1")
