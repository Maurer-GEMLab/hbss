# HBSS Software Development & Testing Plan

**Companion to:** `HBSS-plan.md` (the science/methods plan)
**Scope:** repository structure, interfaces, testing, engineering practice, and a staffing-aware schedule
**Status:** Draft for Year 1 kickoff. Revise with the postdoc in week 1.

**Assumptions made in this document** (correct them if wrong):
- Year 1 runs **1 Sept 2026 – 31 Aug 2027**.
- Postdoc starts at or near the beginning of Year 1.
- PhD student starts **January 2027 at the earliest**, possibly not until Sept 2027, and is class-loaded through 2027 with limited research availability until Year 2–3.
- Nothing on the Year 1 critical path may assume a student exists (§5.1).
- Time-series estimation and its corrections are written in-project rather than taken from an external package (§1.7).

---

## 1. Repository strategy

### 1.1 What monorepo means here

One Git repository containing **two installable Python packages** plus a non-installed research sandbox:

- `hbss` — the library. Pure inference, I/O, simulation, evaluation. Depends on JAX/NumPyro. **Never imports torch.**
- `hbss-learned` — LNCM and LUEP training code. Depends on torch, and on `hbss` for shared data types. Nothing in `hbss` depends on it.
- `experiments/` — dated, disposable research code. Not installed, not tested, not reviewed.

This is deliberately only two packages. Splitting `io`, `kalman`, `mcmc` etc. into separate distributions would be ceremony without benefit at this team size; they share a dependency stack and always version together. The split that *does* earn its keep is JAX-side vs torch-side, because those have genuinely incompatible install stories (CUDA builds, wheel sizes, GPU memory) and because it is the boundary that must survive if a learned component fails.

### 1.2 Layout

```
hbss/                                # one Git repository
├── pyproject.toml                   # workspace root: dev tooling, shared config
├── pixi.toml / environment.yml      # conda-forge stack (ISCE3, Dolphin, GDAL…)
├── uv.lock                          # single lockfile across both packages
├── README.md
├── CONTRIBUTING.md                  # how to work here; read before first PR
├── packages/
│   ├── hbss/
│   │   ├── pyproject.toml           # numpy scipy xarray zarr jax numpyro
│   │   ├── src/hbss/
│   │   │   ├── io/                  # NISAR / Sentinel-1 / GNSS / RAiDER ingest
│   │   │   ├── state/               # latent state definitions
│   │   │   ├── observation/         # per-sensor observation operators
│   │   │   ├── kalman/              # inner-loop KF/RTS (JAX) + dense reference
│   │   │   ├── mcmc/                # outer loop (NumPyro) + discrete Gibbs
│   │   │   ├── noise/               # NoiseModel protocol + parametric models
│   │   │   ├── unwrap/              # k-states, UnwrapProposal protocol, uniform
│   │   │   ├── priors/              # plate motion, GMRF spatial coupling
│   │   │   ├── baseline/            # least-squares time-series estimator
│   │   │   ├── simulate/            # synthetic scene generator  ← high leverage
│   │   │   ├── evaluate/            # metrics, coverage, calibration, SBC harness
│   │   │   ├── postprocess/         # velocity, strain rate, slip deficit
│   │   │   ├── viz/
│   │   │   ├── config.py            # typed run configuration
│   │   │   └── ledger.py            # run provenance record
│   │   └── tests/
│   └── hbss-learned/
│       ├── pyproject.toml           # torch, hbss
│       ├── src/hbss_learned/
│       │   ├── data/                # training-pair construction, splits
│       │   ├── lncm/                # Learned Noise Covariance Model
│       │   ├── luep/                # Learned Unwrapping-Error Proposal
│       │   ├── train/               # loops, schedules, checkpointing
│       │   └── export/              # → artifacts consumed by hbss
│       └── tests/
├── experiments/                     # sandbox; see §1.4
├── tests/integration/               # cross-package, slow
├── benchmarks/
├── docs/
│   ├── decisions/                   # ADR-0001.md, ADR-0002.md, …
│   └── onboarding.md
└── .github/workflows/
```

### 1.3 The one rule that makes this work

**`hbss` must run, and pass its entire test suite, in an environment where torch is not installed.**

This is not a style preference. It is what lets you delete `lncm/` in Year 2 without touching the baseline, and it is what keeps the Year-1-completes-without-ML risk posture from §3.6 of the science plan actually true rather than aspirational.

Enforce it mechanically with a CI job that installs *only* `packages/hbss` into a clean environment and runs its tests. Any accidental `import torch` fails the build immediately, with an obvious error. This is more robust than a linter rule and takes ten minutes to set up.

### 1.4 `experiments/` — the flexibility valve

You said you need to stay flexible until you know what works. That flexibility should be *localized*, not spread through the library as half-finished abstractions.

- Convention: `experiments/YYYY-MM-initials-slug/` (e.g. `2026-10-jm-numpyro-scan-spike/`).
- Each directory has a `README.md` with five lines: **question / what I did / result / verdict / where the code went**.
- Excluded from `ruff`, `mypy`, `pytest`, and code review in the tool configs. Push straight to `main`.
- Nothing in `packages/` may import from `experiments/`. (Same CI trick: it isn't installed, so it can't.)

**Graduation rule.** Code moves `experiments/ → packages/` when, and only when, it has (a) a test, (b) a stable interface, and (c) someone other than the author has read it. Review graduations once a month in a standing meeting. This gives the postdoc and student full freedom to thrash on methods while the library stays something you can hand to a stranger.

### 1.5 Interfaces — pin the seams, let the implementations churn

The methods are uncertain; the *shape* of the plug-in points is not. Define these in Phase 1 and hold them stable.

```python
# hbss/noise/protocol.py
class NoiseModel(Protocol):
    def covariance(self, obs: ObsBatch, cov: Covariates) -> ObsCovariance: ...


# ObsCovariance is a small sum type: Diagonal(var) | Dense(matrix)
# — returning this type from day one makes §6.2 (diagonal → full R) an
#   additive change rather than a breaking one.
```

```python
# hbss/unwrap/protocol.py
class UnwrapProposal(Protocol):
    def propose(self, key, k: Array, feats: CCFeatures) -> tuple[Array, Array]:
        """Return (k', log q(k|k') - log q(k'|k))."""
```

**Return the log proposal ratio from the very first implementation**, even though it is identically zero for the symmetric uniform proposal. Retrofitting it when the LUEP arrives in Year 3 is precisely where a silent Metropolis–Hastings bug gets introduced, and that bug produces plausible-looking, wrong posteriors.

Implementations then stack up behind each seam: `ConstantNoise`, `CoherenceNoise` (the `1/γ²` baseline), `LearnedNoise`; `UniformProposal`, `LocalRandomWalkProposal`, `LUEPProposal`. Selecting among them is a config field, so an experiment is a config diff — not a branch, not a fork.

### 1.6 How the learned components integrate (recommendation)

**Export to artifacts; do not call torch from the sampler.**

- LNCM: run offline over the tile store, **write predicted `R` fields into Zarr** alongside the observations. The JAX inner loop reads an array. Uncertainty over `R` (the deep-ensemble spread from §2.2) is written as additional bands and consumed as a hyperprior.
- LUEP: it is a *fixed* proposal (your §3.4 says so). Precompute, per interferogram, a table of per-connected-component candidate probabilities. The MCMC reads the table.

This removes the entire class of problems around torch and JAX sharing a process and a GPU, keeps the core dependency-free, makes runs reproducible from a versioned artifact, and costs you nothing you actually need. A live in-process model can be added later behind the same protocol if a use case demands it.

### 1.7 Time-series estimation is written in-project

`hbss/baseline/` implements least-squares time-series estimation and its corrections directly, rather than depending on an external toolchain. The problem is small — a design matrix over interferogram pairs, a weighted least-squares solve, and a handful of correction terms — and the dependency, environment, and licensing overhead of an external package is out of proportion to it.

Scope, in the order it should be built:

1. Network inversion: build $\mathbf{G}$ from the pair list; weighted least-squares for cumulative displacement; **explicit** handling of rank deficiency from disconnected network components (minimum-norm or Tikhonov), reported rather than silently regularized away.
2. Corrections: tropospheric delay from RAiDER ZTD fields; DEM error via the $B_\perp$-proportional term; orbital and long-wavelength ramps; reference-point selection.
3. Velocity fitting with formal covariances.

This module pays for itself three times: it is the HBSS warm start, it is the quantitative comparison baseline for real-data validation, and building it early is the cheapest way to exercise the I/O layer and the observation operators against real data before the state-space machinery sits on top of them. It also means that if the inner loop hits trouble, the project still has a working time-series estimator to fall back on.

Two things to watch. First, it shares the observation operators with the HBSS — the design matrix is the same $\mathbf{H}$ without the state-space dynamics — so build it *on* `hbss/observation/` rather than beside it, or the two will drift apart and the comparison will stop being apples-to-apples. Second, this is a well-trodden problem with published reference results; validate against Berardino et al. (2002) / Schmidt & Bürgmann (2003) formulations and a synthetic network with known truth before trusting it as a baseline.

---

## 2. Testing strategy

Bayesian inference code fails silently. It produces smooth, plausible, publishable posteriors that are wrong. Ordinary unit tests will not catch this, so the test plan below is weighted toward two techniques that will.

### 2.1 Tiers

| Tier | What | Runtime | When |
|---|---|---|---|
| **T0** Unit | shapes, dtypes, algebra, pure functions | < 60 s | every push |
| **T1** Statistical | dense reference, SBC, property tests | ~5 min | every PR |
| **T2** Integration | small end-to-end synthetic scenes | ~20 min | nightly |
| **T3** Benchmark & real-data | 256×256×30 tile, baseline comparison, GNSS residuals | hours | weekly + pre-release |

### 2.2 The two tests that matter most

**(a) Dense-solve reference for the Kalman/RTS inner loop.** For a linear-Gaussian state-space model the full joint posterior over `s_{1:T}` is a single multivariate Gaussian. Build its block-tridiagonal precision matrix and solve it directly with SciPy for small `T`. The RTS smoother means and covariances must match to ~1e-9, and the filter's marginal likelihood must match `log N(y; μ, Σ)` computed from the dense joint.

Keep this reference implementation **permanently in the test suite** — it is slow, obviously correct, and is the tool you will reach for every time the fast path misbehaves. This is a better answer to open question §6.1 than "a built-in fallback for prototyping."

**(b) Simulation-based calibration (SBC) for the whole posterior.** Draw `θ ~ p(θ)`, then `y ~ p(y|θ)`, run inference, and record the rank of the true `θ` among posterior draws. Over many replications those ranks must be uniform. Non-uniformity is a proof of a bug in the model, the sampler, or the likelihood.

Run SBC on the single-pixel model at T1 and on the tile model at T2. It is simultaneously your regression test and the evidence a reviewer will demand for the calibrated-uncertainty claims in §1.6. (Talts et al. 2018, arXiv:1804.06788 — worth adding to the Week 2 reading list.)

Supplementary: property-based tests via `hypothesis` for the algebra (covariances stay symmetric positive-definite; information-filter and covariance-filter forms agree; a zero-noise system reproduces its input exactly), and a Geweke joint-distribution test for the Metropolis-within-Gibbs sweep over `k` once that exists.

### 2.3 Integration tests

10×10 pixels × 10 epochs, injected truth, in CI:
- 95% credible intervals cover the truth for 90–98% of pixels;
- `R̂ < 1.01` and ESS above a floor on all hyperparameters;
- velocity RMSE below a fixed threshold.

Add a standing **identifiability test** from the §1.5 pitfalls: a synthetic scene with elevation-correlated *deformation* and zero troposphere. The framework must not absorb it into the tropospheric state. Make this a test that fails the build, not a figure someone remembers to make.

### 2.4 Testing the ML components

- **Overfit-one-batch.** Training must drive loss to ~0 on a single batch. Catches broken data pipelines and misaligned labels, which is where most ML bugs live.
- **Leakage guard as a unit test.** Train/val/test splits must be blocked in **space and time** — assert a minimum separation between split members. InSAR pixels near the same GNSS station on adjacent dates are near-duplicates; an unblocked random split will report excellent held-out performance that is entirely fictitious. Test the split function itself.
- **Calibration as a gate, not a finding.** Held-out ECE below a threshold and coverage inside a band are pass/fail conditions on a model artifact. A miscalibrated model does not get exported.
- **Model cards.** Every exported artifact carries training-data manifest hash, code SHA, config, and validation metrics.

### 2.5 Reproducibility ledger

Every run writes `run.json`: git SHA + dirty flag, fully resolved config, input data manifest hashes, all RNG seeds, package versions, host/GPU, wall time. Build this in **Phase 0**, before any science happens — retrofitting provenance after eighteen months of results is miserable and never quite works.

---

## 3. Engineering practice

**Environments.** Two: `core` (JAX/NumPyro) and `learned` (torch). Do not merge them; co-installing CUDA-enabled JAX and torch is a recurring time sink with no upside given the export-artifact design in §1.6. The geodesy stack (ISCE3, Dolphin, GDAL) is conda-forge-only, so the base is conda-forge with pip/uv on top — **`pixi` is the cleanest tool for that combination** (one lockfile spanning conda and PyPI, multi-environment support). Fall back to `environment.yml` + `uv` if pixi causes friction; decide in week 1 and record an ADR.

**Data.** Never in Git. `packages/hbss/src/hbss/data/registry.yaml` maps logical dataset names to URI, checksum, size, and provenance; a `hbss-fetch` command populates `$HBSS_DATA_ROOT`. Test fixtures ≤ 5 MB and generated by `hbss.simulate` wherever possible. **The institutional storage decision is a Phase 0 blocker** — the postdoc cannot start real-data work without it.

**CI (GitHub Actions).** lint (`ruff`) · typecheck (`mypy`, `packages/` only) · T0+T1 on Python 3.11 and 3.12 · **core-isolation job** (§1.3) · docs build · nightly T2 · weekly T3 on a self-hosted GPU runner if you have one. Keep the PR path on hosted runners so a dead lab machine never blocks a merge.

**Review.** `packages/**` requires a PR, one approving review, and green CI. `experiments/**` requires nothing. Weekly 60-minute code-and-results meeting; monthly graduation review.

**Decision records.** `docs/decisions/ADR-NNNN.md`, ~one page each: context, decision, alternatives, consequences. This is how the student arriving in 2027 learns why the inner loop is written the way it is, and how you write the methods paper's software section without archaeology.

---

## 4. Schedule

### Phase 0 — Foundations · Sept 2026 · PI + postdoc

No science. Deliverables: repo and workspace scaffolded; both environments reproducible on the target machine; CI green including the core-isolation job; config system and run ledger working; data storage decided and one real Sentinel-1 stack (southern California) downloaded; ADRs written for all six open questions in §6 of the science plan.

Postdoc reads the Week 1–4 list in parallel.

*Exit:* `pytest` passes and `hbss run configs/toy.yaml` produces a run directory with a complete provenance ledger.

### Phase 1 — Vertical slice, single pixel · Oct–Nov 2026

**Start with a timeboxed two-week spike on the riskiest integration, before any I/O work**: NUTS over `θ` with a JAX Kalman marginal likelihood embedded in a NumPyro model, on a one-pixel toy problem. The continuous/discrete composition (NUTS over `θ`, Gibbs over `k`) is the single assumption most likely to force an architecture change, and you want to discover that in October 2026, not April 2027.

Then: state and observation operators for L-band, C-band, and GNSS; parametric noise models; the dense reference test; SBC on the single-pixel model.

*Exit:* synthetic test 1 from §1.6 passes with calibrated coverage. ADR records whether the NumPyro composition works, or whether plan B (hand-rolled Gibbs outer loop, or empirical Bayes on the marginal likelihood via Optax) is needed.

### Phase 2 — Baseline estimator, tiles, discrete states · Dec 2026–Feb 2027

Open with `hbss/baseline/` (§1.7). It is roughly two weeks, it forces the I/O layer and observation operators into a usable state against real data, and it delivers both the warm start and the comparison baseline before anything depends on them.

Then: `vmap` over pixels; Zarr tile I/O and chunk strategy; Metropolis-within-Gibbs over `k` with uniform proposals; benchmark harness.

**Build the compute budget model here.** Measure the cost of one tile posterior, extrapolate to the full plate-boundary application, and compare against the compute you actually have. If the answer is infeasible it will stay infeasible, and the architectural response (empirical Bayes on hyperparameters, coarser tiling, two-stage estimation) is far cheaper to make now than after spatial coupling is built on top.

*Exit:* a 256×256×30 tile runs end to end; documented go/no-go on the inference architecture.

### Phase 3 — Simulator v0 and spatial coupling · Mar–May 2027

The **synthetic scene generator** (`hbss.simulate`) is the highest-leverage single item in the project: it is required for synthetic test 2, and it is the LUEP's entire training-data supply. The science plan builds it implicitly in Year 2 Q4, which puts the LUEP's hardest dependency last and least-staffed.

It is built in **two tiers**, and the split is what keeps the infrastructure off the student's timeline — see §5.1.

- **`v0` — postdoc or PI, ~2 weeks, this phase.** Known displacement field, spatially correlated Gaussian noise via the same GMRF machinery used in the priors, injected $2\pi$ errors on synthetic connected components. No claim to physical realism. Sufficient for synthetic test 2, the identifiability control test, and CI integration fixtures.
- **`v1` — student, from Phase 4 onward.** Physically-based decorrelation, DEM-driven topography, unwrapper-in-the-loop labeling. Replaces the `v0` implementation behind the same interface.

Also this phase: GMRF spatial coupling at the hyperparameter level, and real-data test 1 on southern California — HBSS against `hbss/baseline/` on identical inputs, with GNSS as the independent truth for both.

### Phase 4 — v0.1 · Jun–Aug 2027

Sphinx docs; tutorial notebooks auto-executed in CI; Hispaniola leave-one-out (real-data test 2); tag **v0.1**; methods paper skeleton.

### Year 2 onward

LNCM first, and **start with training-pair construction, not the network**. The bottleneck is assembling 10⁴–10⁵ labeled residuals with defensible spatial and temporal blocking; the CNN itself is about a week. Postdoc leads HBSS integration. LUEP begins once simulator `v1` is producing labeled scenes at volume.

---

## 5. The PhD student

Two constraints are in tension: the student needs three papers' worth of substantial, ownable work, and the critical infrastructure cannot wait for a student who is class-loaded through 2027 and not fully research-available until Year 2 or 3. The resolution is structural rather than a matter of scheduling harder.

### 5.1 The stub policy

**Every student-owned component has a minimal, working `v0` built by the PI or postdoc before the student takes it over.** The `v0` is deliberately unambitious — it exists to unblock everything downstream, not to be good — and it establishes the interface that the student's `v1` will satisfy.

This makes the arrangement work in both directions. The project never blocks on student availability, so a semester lost to quals or coursework costs momentum rather than a milestone. And the student inherits a working system with a defined interface and existing tests, which is a far better starting point than an empty directory and is not experienced as being handed someone else's scraps: `v0` is a placeholder everyone understands to be temporary, and replacing it is unambiguously the student's contribution.

The simulator is the first instance (§Phase 3). The same pattern applies to the LUEP proposal (`UniformProposal` exists from Year 1; the LUEP replaces it) and to any later student-owned component.

### 5.2 Dissertation arc — three papers

Assumes a January 2027 start. Coursework dominates through 2027; the first full-time research block is summer 2027.

**Paper 1 — a physically-based InSAR interferogram simulator and a systematic characterization of unwrapping-error statistics.**
*Work: summer 2027 – summer 2028. Submit ~Y2 Q4.*

Not "we wrote a simulator," which is thin. The paper is the *study the simulator enables*: unwrapping-error rates and spatial structure as a function of coherence, topography, baseline, and unwrapper configuration, quantified over a large labeled corpus that nobody currently has. The deliverables are an open simulator, a public labeled benchmark, and error statistics for SNAPHU-class unwrappers.

This is the ideal first project for a class-loaded student. It is modular and tolerant of intermittent effort, it is testable against theory (Zebker & Villasenor) rather than requiring the full pipeline to work, it needs no NISAR data to exist yet, and it is off the critical path in the sense that `v0` already covers the project's needs. It is also independently useful and citable **even if the LUEP fails entirely** — which matters, because it means the student's first paper does not carry the project's largest technical risk.

**Paper 2 — learned proposal distributions for discrete unwrapping-error inference (the LUEP).**
*Work: Y3 Q1 – Y3 Q3. Submit ~Y3 Q4.*

The student owns this by construction: they built the training data, so they understand its failure modes better than anyone. Methods paper, evaluated on both the calibration criteria and the MCMC mixing speedup. A well-characterized negative result here is publishable given the benchmark from paper 1 — the question "can a learned proposal beat uniform for this problem, and under what conditions" is worth answering either way.

**Paper 3 — application, scoped by the student.**
*Work: Y3 Q4 onward, extending past the grant period.*

Nominally the plate-boundary application with a specific tectonic question attached (Hispaniola slip deficit and strain partitioning is the obvious candidate), but this one should be **left genuinely open for the student to propose**. By late Year 3 they will have spent two years inside the framework and will have better ideas about its unexploited capabilities than anyone did in 2026. Committing the third paper now would be planning fiction, and a dissertation whose final chapter the student chose themselves is a materially better dissertation.

Note the grant ends in August 2029 and a January 2027 start implies a 2031–2032 defense. Papers 1 and 2 fall inside the award; paper 3 straddles the end and will need a funding plan — flag this at the student's first committee meeting rather than discovering it in Year 4.

### 5.3 Ramp expectations

| Period | Availability | Focus |
|---|---|---|
| Jan–May 2027 | ~10 h/week, coursework dominant | Onboarding, reading list, small reviewed PRs, take handoff of simulator `v0` |
| Summer 2027 | Full time — first real block | Simulator `v1` core physics; this is the make-or-break block for paper 1 |
| Y2 (Sept 2027–Aug 2028) | ~half, plus quals/proposal | Simulator `v1` completion, benchmark corpus, paper 1 submission |
| Y3 onward | Full | LUEP, paper 2, then paper 3 |

**Write `docs/onboarding.md` during Phase 2 (Dec 2026–Feb 2027), while the design decisions are still fresh** — not in the week before the student arrives. It should get a new person from clone to a passing test suite to a first merged PR in under three days. The ADRs in `docs/decisions/` are the other half of this: they are how the student learns *why* the inner loop is written the way it is without having to ask.

Give the student a real reviewed PR in their first month, however small. A first contribution that merges is worth more than three weeks of reading.

### 5.4 Staffing contingencies

**No student until Sept 2027:** postdoc holds the critical path unchanged. PI builds simulator `v0` in Phase 3 as planned — it was never on the student's timeline — and `v1` slips by two quarters. LNCM slips one quarter. LUEP moves to late Year 3 with the uniform-proposal fallback as the honest published comparison.

**No student at all:** invoke the descope ladder. Paper 1's benchmark study is the piece to preserve if anything can be — it is the most self-contained and has value independent of the rest.

### 5.5 Descope ladder (cut in this order)

1. Full covariance `R` → diagonal only.
2. LUEP live proposal → precomputed candidate tables from a smaller, simpler classifier.
3. LUEP → uniform proposals, published as a documented negative result on learned proposals.
4. Full plate-boundary application → two representative subregions.

**Never cut:** the reproducibility ledger, the dense-reference and SBC tests, and the v0.1 baseline release. These are what make the project defensible if the learned components do not pan out.

---

## 6. Risk register

| # | Risk | Mitigation | Phase |
|---|---|---|---|
| 1 | NumPyro/JAX composition of continuous + discrete inference doesn't work as assumed | Timeboxed spike first thing in Phase 1; plan B recorded in an ADR | 1 |
| 2 | Full-scene MCMC is computationally infeasible | Compute budget model before spatial coupling is built | 2 |
| 3 | LUEP has no labeled training data | Simulator `v0` in Year 1; student's `v1` from Year 2 | 3 |
| 4 | JAX/torch/CUDA environment conflicts | Package split + offline artifact export; never both in one process | 0 |
| 5 | Staffing gap | Stub policy (§5.1): nothing on the critical path waits on the student | all |
| 6 | Troposphere/topography and ionosphere/ramp identifiability | Control-region checks as standing build-failing tests | 1–3 |
| 7 | In-project baseline estimator is subtly wrong, invalidating the comparison | Validate against published formulations + synthetic network with known truth before use as a baseline | 2 |

Risk 7 is the cost of writing the time-series estimator ourselves rather than depending on an established package, and it is worth naming explicitly. The mitigation is cheap — the estimator is small, the reference formulations are published, and a synthetic network with known truth exercises it completely — but it has to actually be done before any HBSS-versus-baseline comparison appears in a manuscript. Rank-deficiency handling on disconnected networks is the specific place to look hardest.

The license risk that previously sat in this table is resolved: implementing time-series estimation in-project removes the only anticipated copyleft exposure. ISCE3, Dolphin, and RAiDER are all expected to be permissively licensed — verify in week 1 and take all three to OSP together with the Apache-2.0 confirmation.

---

## 7. Answers to the open questions in §6 of the science plan

1. **Custom JAX Kalman.** Keep the dense reference implementation permanently in the test suite rather than as a prototyping fallback — it is your debugging instrument, not a stepping stone.
2. **Diagonal `R`.** But have `NoiseModel` return an `ObsCovariance` sum type from day one so adding dense support is additive.
3. **Physics-simulated decorrelation.** Settled by dual use: the same simulator is required for HBSS validation regardless.
4. **Monorepo**, two packages, one-way dependency enforced in CI. See §1.
5. **Apache 2.0**, plus SPDX headers and a `NOTICE` file. Verify ISCE3, Dolphin, and RAiDER licenses alongside it; time-series estimation is written in-project (§1.7), which keeps the dependency surface small and permissive.
6. **GitHub Actions** for T0–T2. One self-hosted GPU runner for T3 nightly/weekly only; never on the PR path.

---

## 8. First two weeks

- [ ] Create the repo; scaffold the two packages and `experiments/`; push a green CI skeleton.
- [ ] Stand up both environments on the target machine and commit the lockfile.
- [ ] Add the core-isolation CI job (§1.3) — it is trivial now and painful later.
- [ ] Write `ledger.py` and `config.py`; wire them into a toy command.
- [ ] Decide data storage; write `registry.yaml` and `hbss-fetch`; pull one SoCal stack.
- [ ] Write ADR-0001 through ADR-0006 from §7 above.
- [ ] Email OSP: Apache 2.0 confirmation + Dolphin/ISCE3/RAiDER license verification.
- [ ] Draft `CONTRIBUTING.md` covering the graduation rule, the review policy, and the stub policy (§5.1).
- [ ] Schedule the standing weekly meeting and the monthly graduation review.
- [ ] Postdoc: Week 1–2 reading list, plus Talts et al. (2018) on SBC.
- [ ] Put a calendar reminder on Dec 2026 to write `docs/onboarding.md` before the student arrives.

---

*Companion to `HBSS-plan.md`. Revise at kickoff.*
