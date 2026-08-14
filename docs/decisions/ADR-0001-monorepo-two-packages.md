# ADR-0001: Monorepo with two installable packages

- **Status:** Accepted
- **Date:** 2026-08-13
- **Deciders:** Maurer (PI). To be reviewed with the postdoc at kickoff.

## Context

The project has three coupled software components — the HBSS inference engine,
the learned noise model (LNCM), and the learned unwrapping-error proposal
(LUEP) — developed by two to three people over three years, with the learned
components arriving in Years 2 and 3 and carrying real risk of not working out.

Two forces pull in opposite directions. The components are tightly coupled and
their interfaces will churn, which argues for keeping them together. But they
have incompatible dependency stacks — JAX/NumPyro versus torch, both wanting
CUDA — and the project's risk posture requires that the baseline framework
remain shippable if either learned component fails, which argues for keeping
them apart.

## Decision

One Git repository containing **two separately-installable packages**:

- `packages/hbss` — inference, I/O, baseline estimator, simulator, evaluation.
  JAX/NumPyro. Never imports torch.
- `packages/hbss-learned` — LNCM, LUEP, training, export. Torch. Depends on
  `hbss`; the reverse dependency does not exist.

Plus `experiments/`, a dated sandbox that is not installed, not linted, not
tested, and not reviewed.

The dependency direction is enforced by a CI job that installs `hbss` alone
into an environment where torch is absent and runs its full test suite.

Learned models integrate by **exporting artifacts, not by being imported**: the
LNCM writes predicted `R` fields into the Zarr tile store, the LUEP writes
per-connected-component candidate probability tables, and the sampler reads
arrays.

## Alternatives considered

**Separate repositories per component.** Genuine isolation, and independent
release cadences if the components ever mature at different rates. Rejected on
coordination cost: at this team size, a change to a shared interface would
become three pull requests, a version pin, and a week of drift, repeatedly,
for the entire project. The isolation benefit is obtainable inside a monorepo
via the package split; the coordination cost is not recoverable.

**One package with optional extras.** Simplest possible thing, and it was
close. Rejected because the torch dependency would then be reachable from the
core distribution's metadata even when unused, CI would install the full ML
stack to test a Kalman filter, and — decisively — there would be no mechanical
way to prove the baseline still stands alone. The check is the point.

**More than two packages** (splitting `io`, `kalman`, `mcmc`, …). Rejected as
ceremony: they share a dependency stack, always version together, and no
consumer wants one without the others. Two packages is the split that tracks a
real boundary. Revisit only if a component acquires an independent user base.

**Live in-process model calls** instead of exported artifacts. Rejected: it
would put torch and JAX in one process contending for one GPU, reintroduce the
torch dependency into the core, and make runs reproducible only against a
mutable model rather than a checksummed artifact. Since the LUEP is a *fixed*
proposal and the LNCM is applied per-pixel offline, live evaluation buys
nothing. Can be added later behind the same protocols if a use case appears.

## Consequences

Makes easy: atomic cross-component interface changes; one clone, one CI, one
issue tracker for onboarding; deleting a learned component without touching the
baseline; installing the framework without a CUDA torch build.

Makes hard: the dependency direction must be actively policed rather than
being a natural consequence of separate repositories — mitigated by the CI job,
which is why that job is not optional. Releasing the two packages on genuinely
independent schedules would take additional work, but nothing needs that now.

Commits us to: the artifact-export integration path, which shapes how the LNCM
and LUEP are designed in Years 2 and 3; and to `experiments/` as the place
where methodological uncertainty is absorbed, rather than the library.

## Revisit if

A learned component acquires users independent of the HBSS framework, or if
the artifact-export path proves inadequate for a use case that matters — for
instance, if a learned proposal must adapt during sampling rather than being
fixed.
