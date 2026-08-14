# Experiments

The research sandbox. Push directly to `main`. Not installed, not linted, not
type-checked, not tested, not reviewed.

This is where you find out whether a method works. It is deliberately free of
process so that finding out stays cheap — the project needs to stay flexible
until the methods are settled, and the way to do that is to localise the
thrashing here rather than spread half-finished abstractions through the
library.

## Convention

```
experiments/YYYY-MM-initials-slug/
    README.md      <- five lines, required
    ...            <- whatever you like
```

Copy [`_template/`](_template/) to start. The `README.md` is the only rule, and
it exists so that in eighteen months someone can tell at a glance whether a
directory contains a dead end or something worth reviving — including you.

## Graduating code

Code moves to `packages/` when it has a test, a stable interface, and someone
other than the author has read it. Reviewed monthly.

Nothing in `packages/` may import from here. It isn't installed, so it can't.

## Index

Keep newest at the top. One line each.

| Directory | Question | Verdict |
|---|---|---|
| _(none yet)_ | | |
