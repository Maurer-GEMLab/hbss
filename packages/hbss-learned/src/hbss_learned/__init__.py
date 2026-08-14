"""Learned components for the HBSS framework: the LNCM and the LUEP.

This package depends on torch and on ``hbss`` (for shared data types). The
dependency runs one way only -- ``hbss`` knows nothing about this package.

Both models are consumed by the inference engine as *exported artifacts*, not
as live imports:

    LNCM  ->  predicted R fields written into the Zarr tile store
    LUEP  ->  per-connected-component candidate probability tables

Every export carries a model card (training-data manifest hash, code SHA,
config, validation metrics), and calibration thresholds are a gate on export
rather than something reported afterwards.

See HBSS-plan.md sections 2 and 3, and HBSS-dev-plan.md sections 1.6 and 2.4.
"""

__version__ = "0.0.0.dev0"
