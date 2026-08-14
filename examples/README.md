# Examples

Tutorial notebooks, one per major workflow. Auto-executed in CI so they cannot
silently rot — a notebook that no longer runs is worse than no notebook.

Planned:

1. Ingest InSAR data and build a tile store.
2. Run the least-squares baseline estimator.
3. Define an HBSS model and run inference.
4. Post-process to a velocity field with uncertainty.
5. Inspect and interpret posterior diagnostics.

Keep them small enough to execute in CI — use `hbss.simulate` output or a
minimal registered dataset, never a full stack.
