# 2026-09-MR-syn-interferogram-generator

**Question.** What were you trying to find out? One sentence, phrased so it can
be answered yes or no, or with a number. Can a pre-built repo for synthetic inteferogram generation, called eq-insar, be retrofitted to meet this project's needs?

**What I did.** Enough that someone could repeat it. Name the data, the config,
the commit if it matters. The noise.py file was modified to include decorrelation noise. Tests were added to check. 

**Result.** What happened. A set of synthetic interferograms with several decorrelation noise types was generated. Noise included that from snow and vegetation, as well as from a custom mask.

**Verdict.** Worth pursuing / dead end / inconclusive, and why. A dead end
written down clearly is worth as much as a success — it stops the next person
spending three weeks rediscovering it. Contuining to retrofit the repo to also include unwrapping errors is worth pursuing.

**Where the code went.** Graduated to `packages/...`, abandoned here, or
superseded by `experiments/...`. Code is inside experiments/2026-09-MR-syn-interferogram-generator