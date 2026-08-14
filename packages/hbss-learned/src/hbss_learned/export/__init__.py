"""Export trained models to artifacts consumed by `hbss`.

The LNCM writes predicted R fields into the Zarr tile store; the LUEP writes
per-connected-component candidate probability tables. `hbss` reads arrays and
never imports torch. Calibration is a gate on export, not a later finding.
"""
