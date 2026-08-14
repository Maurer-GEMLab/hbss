# Architecture decision records

One page per non-obvious decision: context, decision, alternatives,
consequences. Copy [`ADR-0000-template.md`](ADR-0000-template.md).

ADRs are cheap to write and expensive to skip. They are how someone joining in
2027 learns *why* the inner loop is written the way it is without having to ask,
and how the methods paper's software section gets written without archaeology.
Superseded ADRs are marked superseded, never deleted — the reasoning that
turned out to be wrong is often the most useful thing in the file.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-monorepo-two-packages.md) | Monorepo with two installable packages | Accepted |
| 0002 | Environment and dependency management (pixi vs. conda + uv) | **To write, week 1** |
| 0003 | Custom JAX Kalman with a permanent dense reference | **To write, week 1** |
| 0004 | Diagonal `R` behind a covariance sum type | **To write, week 1** |
| 0005 | Physics-simulated decorrelation for LUEP training data | **To write, week 1** |
| 0006 | Apache 2.0, and the in-project time-series estimator | **To write, week 1** |
| 0007 | CI on GitHub Actions; self-hosted runners for tier 3 only | **To write, week 1** |

Decisions 0002–0007 are already resolved in `HBSS-dev-plan.md` section 7; these
ADRs write up the reasoning and are the artifact the postdoc reviews and, where
warranted, argues with.

Expected in Phase 1, once there is evidence:

- Does NUTS over θ compose with the JAX Kalman marginal likelihood and Gibbs
  over **k** in NumPyro? (The Phase 1 spike answers this.)
- Is full-scene MCMC affordable at the plate-boundary scale? (The Phase 2
  compute budget model answers this.)
