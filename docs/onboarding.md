# Onboarding

> **Status: stub.** Write this properly during Phase 2 (Dec 2026 – Feb 2027),
> while the design decisions are still fresh — not in the week before someone
> arrives. Target: clone to passing test suite to first merged pull request in
> under three days.

## Day 1 — get it running

```bash
git clone <repo> && cd hbss
uv sync --all-packages
uv run pytest
```

If that fails, the failure is a bug in this document or in the setup, not in
you. Say so and we will fix it — every new person makes this page better.

TODO: the conda-forge side (`pixi install`), GPU setup, data access, and
`$HBSS_DATA_ROOT`.

## Day 1–2 — read

1. `HBSS-plan.md` sections 0–1 — what the model is and how inference works.
2. `HBSS-dev-plan.md` sections 1–2 — how the code is organised and how it is
   tested.
3. `CONTRIBUTING.md` — the graduation rule, the dependency rule, the stub
   policy.
4. `docs/decisions/` — why things are the way they are.

Then the Week 1 reading list in `HBSS-plan.md` section 5. Segall & Matthews
(1997) and Dalaison & Jolivet (2020) are the two that matter most for
understanding what the inner loop is doing.

## Day 2–3 — make something merge

Pick up a small issue and open a pull request. A first contribution that merges
is worth more than three weeks of reading, and it exercises the whole
workflow — branch, test, review, CI — while the stakes are low.

TODO: maintain a `good-first-issue` label with two or three genuinely small,
genuinely useful items at all times.

## Things that will not be obvious

TODO, but at minimum:

- Why the dense reference implementation of the smoother exists and is slow on
  purpose.
- Why `UnwrapProposal.propose` returns a log ratio that is always zero right
  now.
- Why the learned models are never imported.
- What `experiments/` is for and why nothing in it is reviewed.
