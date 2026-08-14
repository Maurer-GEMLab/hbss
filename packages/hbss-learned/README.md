# hbss-learned

Learned components for the HBSS framework: the Learned Noise Covariance Model
(LNCM) and the Learned Unwrapping-Error Proposal (LUEP).

Torch. Depends on `hbss`; `hbss` does not depend on this.

Models reach the inference engine as **exported artifacts** — predicted noise
fields written into the tile store, per-component candidate probability tables
read by the sampler — never as live imports. Calibration is a gate on export.

Part of the [HBSS monorepo](../../README.md).
