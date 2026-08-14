# Benchmarks

Tier 3: performance regression and real-data comparison. Runs weekly and before
every release — never on the pull-request path, so that a dead lab machine
cannot block a merge.

Planned, from `HBSS-dev-plan.md` section 2.1:

- **Representative tile.** 256×256 pixels, 30 epochs. Track wall clock and peak
  memory over time; a regression is a bug report, not a curiosity.
- **Compute budget model** (Phase 2). Cost of one tile posterior, extrapolated
  to the full plate-boundary application. This one is load-bearing: if the
  extrapolation is infeasible, the architectural response is far cheaper before
  spatial coupling is built on top than after.
- **Real-data regression.** Fixed small stack, golden outputs with tolerance,
  HBSS against `hbss.baseline` on identical inputs with GNSS as independent
  truth for both.
