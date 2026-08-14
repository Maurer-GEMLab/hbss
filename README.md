# HBSS

Hierarchical Bayesian state-space estimation of surface displacement time
series from multi-sensor InSAR.

Displacement is estimated jointly with tropospheric, ionospheric, and
unwrapping-error latent states, with hyperparameters estimated hierarchically
rather than fixed. The output is a posterior over displacement time series with
calibrated uncertainty at each pixel.

> **Status: pre-alpha.** This repository is a skeleton. Year 1 begins September
> 2026 and nothing here computes anything yet. Directory stubs describe what
> will live in them.

## Start here

| | |
|---|---|
| [`HBSS-plan.md`](HBSS-plan.md) | The science: model, inference strategy, validation plan, references, reading list |
| [`HBSS-dev-plan.md`](HBSS-dev-plan.md) | The software: repository structure, interfaces, testing, schedule, staffing |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to work in this repository |
| [`docs/decisions/`](docs/decisions/) | Architecture decision records — why things are the way they are |

New to the project? Read `HBSS-plan.md` sections 0–1 and `HBSS-dev-plan.md`
sections 1–2, then the Week 1 reading list in `HBSS-plan.md` section 5.

## Layout

```
packages/hbss/            the library — JAX/NumPyro. Never imports torch.
packages/hbss-learned/    LNCM and LUEP — torch. Depends on hbss, not vice versa.
experiments/              dated research sandbox. Not installed, not reviewed.
tests/integration/        cross-package, slow
docs/  benchmarks/  examples/
```

Two packages in one repository. The split that matters is JAX-side versus
torch-side: they have genuinely incompatible install stories, and it is the
boundary that has to survive if a learned component fails.

**The dependency arrow points one way.** `hbss` must pass its entire test suite
in an environment where torch is not installed. A CI job (`core-isolation`)
enforces this by actually installing it that way, so a violation fails the
build rather than relying on anyone remembering the rule. This is what lets the
learned components be removed without touching the baseline.

Learned models reach the inference engine by **exporting artifacts, not by
being imported**: the LNCM writes predicted noise fields into the tile store,
the LUEP writes per-component candidate probability tables, and the sampler
reads arrays.

## Development

```bash
uv sync --all-packages          # both packages plus dev tooling
uv run pytest                   # tiers 0 and 1
uv run ruff check . && uv run mypy
```

The conda-forge stack (ISCE3, Dolphin, GDAL) is managed separately — see
`pixi.toml`. Environment tooling is a week-one decision; see ADR-0002.

## License

Apache 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
