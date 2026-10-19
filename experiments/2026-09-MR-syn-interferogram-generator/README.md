# 2026-09-MR-syn-interferogram-generator

**Question.** What were you trying to find out? One sentence, phrased so it can
be answered yes or no, or with a number. I wanted to see if I could retrofit two pre-built synthetic interferogram generators to meet the generation needs of this project.

**What I did.** Enough that someone could repeat it. Name the data, the config,
the commit if it matters.

**Result.** What happened. Include the number, not just the direction.

**Verdict.** Worth pursuing / dead end / inconclusive, and why. A dead end
written down clearly is worth as much as a success — it stops the next person
spending three weeks rediscovering it.

**Where the code went.** Graduated to `packages/...`, abandoned here, or
superseded by `experiments/...`.


TODO: Implement eq-insar in this notebook, generate some basic interferograms, and then try to modify the repo to include decorrelation noise generation.



Some Notes: 

Atmospheric noise creates phase noise (delayed signal). Vegetation creates coherence loss (scrambled phase). 
