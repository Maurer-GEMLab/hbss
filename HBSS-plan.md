# HBSS Implementation Plan

**Project:** Hierarchical Bayesian state-space framework for multi-sensor InSAR time-series estimation
**Author:** PI Maurer (planning notes)
**Status:** Revised post-award (August 2026). Revise again during Year 1 kickoff.

This document is a self-contained quick-start guide for picking up the three software components of the project as quickly as possible once funding is in hand. It is intended for the PI's reference and for handoff to the postdoctoral researcher at the start of Year 1.

**Companion document:** `HBSS-dev-plan.md` covers repository structure, interfaces, testing strategy, engineering practice, the phase-by-phase schedule, and the student's PhD arc. This document covers the science and methods; that one covers how the software gets built. Where the two overlap, the development plan is the more detailed and more current.

---

## 0. Overview and Development Order

The project has three coupled software components:

1. **Hierarchical Bayesian State-Space framework (HBSS)** — the core inference engine. Estimates displacement time series jointly with troposphere, ionosphere, and unwrapping-error latent states. Inner loop is a Kalman/RTS smoother per pixel; outer loop is MCMC over hyperparameters and discrete states.
2. **Learned Noise Covariance Model (LNCM)** — a neural-network regressor that maps per-pixel covariates (coherence, topography, perpendicular baseline, NWP troposphere) to the observation-noise covariance $R_t$ used in the HBSS inner loop.
3. **Learned Unwrapping-Error Proposal (LUEP)** — a probabilistic U-Net that proposes candidate unwrapping-error corrections per connected component, used as a data-driven proposal distribution in the HBSS outer-loop MCMC.

**Implementation order should follow risk management, not the conceptual ordering above.** Build the HBSS framework first with simple parametric noise models and naive MCMC. Add the learned components as upgrades in Year 2. This way the project always has a defensible, complete baseline that can stand on its own if the learned components encounter difficulties.

Two ordering decisions follow from this principle and are worth stating explicitly, because both differ from the obvious reading of the component list:

- **The riskiest integration is spiked first, not last.** Whether NUTS over $\boldsymbol{\theta}$ composes cleanly with a JAX Kalman marginal likelihood and Gibbs over the discrete $\mathbf{k}$ is the single assumption most likely to force an architectural change. It is tested on a one-pixel toy problem in Year 1 Q1, before any I/O work.
- **The synthetic scene simulator moves from Year 2 into Year 1.** It is required for the synthetic validation in §1.6 *and* it is the LUEP's entire training-data supply (§3.3). Building it late puts the LUEP's hardest dependency last and least-staffed. It is built in two tiers: a minimal `v0` by the postdoc/PI in Year 1 Q3, sufficient for HBSS validation, and a physically-based `v1` by the PhD student thereafter (see `HBSS-dev-plan.md` §5).

```
Year 1, Q1:    Foundations (repo, environments, CI, provenance);
               NumPyro/JAX composition spike; single-pixel vertical slice
Year 1, Q2:    Least-squares baseline estimator; tiled inner loop;
               outer loop with naive discrete MCMC; compute budget model
Year 1, Q3:    Simulator v0; spatial coupling priors; real-data test 1
Year 1, Q4:    Documentation, Hispaniola cross-validation, v0.1 release
Year 2, Q1–Q2: LNCM training-pair construction + Midwest pretraining;
               simulator v1 begins (student)
Year 2, Q3:    LNCM Caribbean fine-tuning + HBSS integration (v0.5)
Year 2, Q4:    Simulator v1 complete; LUEP training begins (v1.0)
Year 3, Q1–Q2: LUEP training, validation, HBSS integration (v1.5)
Year 3, Q3–Q4: Full plate-boundary application, validation, manuscripts (v2.0)
```

**Repository structure.** Resolved at kickoff: a monorepo containing two separately-installable packages, with a one-way dependency rule enforced in CI (`hbss` must pass its full test suite in an environment where torch is not installed). Full rationale, layout, and the `experiments/` sandbox policy are in `HBSS-dev-plan.md` §1. Summary:

```
hbss/                            # one Git repository
├── packages/
│   ├── hbss/                    # library: JAX/NumPyro, no torch
│   │   └── src/hbss/
│   │       ├── io/  state/  observation/  kalman/  mcmc/
│   │       ├── noise/           # NoiseModel protocol + parametric models
│   │       ├── unwrap/          # k-states, UnwrapProposal protocol
│   │       ├── priors/  postprocess/  viz/
│   │       ├── baseline/        # least-squares time-series estimator (§1.4)
│   │       ├── simulate/        # synthetic scene generator (§3.3)
│   │       └── evaluate/        # metrics, coverage, calibration, SBC
│   └── hbss-learned/            # torch: LNCM, LUEP, training, export
├── experiments/                 # dated research sandbox; not installed
├── tests/integration/  benchmarks/  docs/  examples/
└── .github/workflows/
```

The learned components integrate by **exporting artifacts, not by being imported**: the LNCM writes predicted $\mathbf{R}$ fields into the Zarr tile store offline, and the LUEP writes per-connected-component candidate probability tables. The inference code reads arrays. This keeps torch out of the sampler entirely (§2.6, §3.4).

---

## 1. Hierarchical Bayesian State-Space Framework (HBSS)

### 1.1 Summary

Estimate the latent displacement time series at each pixel $u(\mathbf{x}, t)$, jointly with latent noise states (troposphere $\tau(\mathbf{x}, t)$, ionosphere $\iota(\mathbf{x}, t)$ where applicable, and unwrapping-error states $\kappa(\mathbf{x}, t, c)$ where $c$ indexes connected components), given multi-sensor InSAR phase observations (NISAR L-band, Sentinel-1 C-band, NISAR S-band where available), GNSS displacements, NISAR's standard correction layers, and weather-model-derived ZTDs.

Hyperparameters (smoothing strengths, noise scales, spatial correlation lengths) are estimated hierarchically rather than fixed. Output is a posterior distribution over displacement time series with calibrated uncertainty intervals at each pixel.

### 1.2 Mathematical Structure

**State equation (continuous latent variables, per pixel):**

$$
\mathbf{s}_t = \mathbf{F}_t \mathbf{s}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(0, \mathbf{Q}_t(\boldsymbol{\theta}))
$$

where $\mathbf{s}_t = [u_t, \tau_t, \iota_t]^T$ is the continuous latent state and $\boldsymbol{\theta}$ are the hyperparameters (smoothing strengths, noise scales, etc.).

For the displacement component, a near-random-walk plus interseismic velocity:

$$
u_t = u_{t-1} + v \cdot \Delta t + w^{(u)}_t, \quad w^{(u)}_t \sim \mathcal{N}(0, q_u(\boldsymbol{\theta}) \cdot \Delta t)
$$

where $v$ is the interseismic velocity (itself a parameter with a truncated-Gaussian prior informed by plate motion). For tropospheric and ionospheric states, similar dynamics with appropriate temporal correlation scales.

**Observation equation (per sensor):**

$$
\mathbf{y}_t^{(s)} = \mathbf{H}_t^{(s)} \mathbf{s}_t + 2\pi \cdot \mathbf{k}_t^{(s)} + \mathbf{v}_t^{(s)}, \quad \mathbf{v}_t^{(s)} \sim \mathcal{N}(0, \mathbf{R}_t^{(s)})
$$

where $s$ indexes the sensor (L-band, C-band, S-band, GNSS), $\mathbf{k}_t^{(s)}$ are integer unwrapping-error states for connected components in observation $t$ of sensor $s$, and $\mathbf{R}_t^{(s)}$ is the sensor-specific observation noise covariance (later parameterized by the LNCM).

The frequency dependence of the troposphere and ionosphere terms is critical: troposphere is approximately frequency-independent in $\mathbf{H}_t^{(s)}$, while ionosphere scales as $1/f^2$ — this is what enables the framework to disentangle them from observations at different frequencies.

**Hyperprior structure:**

$$
p(\boldsymbol{\theta}) = p(q_u) \cdot p(q_\tau, \ell_\tau) \cdot p(q_\iota, \ell_\iota) \cdot \ldots
$$

with weakly-informative priors on noise scales (e.g., Half-Normal) and spatial/temporal correlation lengths (e.g., Inverse-Gamma).

### 1.3 Inference Strategy

**Inner loop (per pixel, fast):** Kalman filter forward pass + RTS smoother backward pass over the continuous state $\mathbf{s}_t$, conditional on current hyperparameters $\boldsymbol{\theta}$ and discrete unwrapping-error states $\mathbf{k}$.

**Outer loop (over hyperparameters and discrete states):** NUTS or HMC over $\boldsymbol{\theta}$; Metropolis-within-Gibbs over $\mathbf{k}$. The unwrapping-error states are connected-component-wise (not per-pixel) which keeps the discrete state space tractable.

**Spatial coupling:** Avoid full pixel-wise spatial coupling in the inner loop (too expensive). Instead, impose spatial coupling at the hyperparameter level: tropospheric and ionospheric noise have Gaussian Markov Random Field structure with spatial correlation length estimated hierarchically. This is computationally tractable and physically motivated (atmospheric fields are spatially smooth).

### 1.4 Implementation Notes

- **Inner loop in JAX.** Per-pixel Kalman/RTS is embarrassingly parallel. JAX with `vmap` over pixels and `jit` over time-step gives 100–1000× speedup over pure NumPy. Use `jaxlib` GPU support where available.
- **Outer loop in NumPyro.** PyMC is more user-friendly but NumPyro is faster on GPU and integrates cleanly with the JAX inner loop. Define the HBSS as a NumPyro model where the inner-loop log-likelihood is a JAX function called from the NumPyro graph.
- **Data layout:** Per-pixel observations stored as a Zarr or HDF5 chunked array. Each chunk = a spatial tile to balance I/O against parallelism granularity. Tile size ≈ 256×256 pixels initially.
- **Reference frame.** Pick one or more reference pixels per region (in stable areas with high GNSS density) and include the reference choice as a nuisance parameter in the outer loop.
- **Initialization.** Warm-start the HBSS from a fast least-squares time-series estimate to accelerate burn-in.
- **Correctness reference.** Keep a slow, dense, obviously-correct implementation of the smoother permanently in the test suite: for a linear-Gaussian model the joint posterior over $\mathbf{s}_{1:T}$ is a single multivariate Gaussian whose block-tridiagonal precision matrix can be solved directly with SciPy for small $T$. The fast JAX path must reproduce it to ~$10^{-9}$. This is the primary debugging instrument for the inner loop, not a prototyping stepping stone.

**The least-squares baseline estimator (`hbss/baseline/`).** Rather than depending on an external time-series package, the project implements its own. The problem is genuinely simple — a design matrix over interferogram pairs, a weighted least-squares (or minimum-norm / Tikhonov-regularized) solve for cumulative displacement, plus a small set of corrections — and writing it ourselves avoids the software and licensing overhead of pulling in a large external toolchain. Scope:

- Network inversion: build $\mathbf{G}$ from the interferogram pair list, solve for cumulative displacement, handle rank deficiency from disconnected network components explicitly rather than silently.
- Corrections: tropospheric delay from RAiDER ZTD fields; DEM error via the $B_\perp$-proportional term; orbital/long-wavelength ramps; reference-point selection.
- Velocity fitting with formal covariances.

This module earns its place three times over: it is the HBSS warm start, it is the primary comparison baseline for the real-data validation in §1.6, and building it in Year 1 Q2 is the cheapest way to write and test the observation operators and the I/O layer against real data. It also means that if the HBSS inner loop hits trouble, the project still has a working time-series estimator.

### 1.5 Pitfalls to Watch For

- **Identifiability between troposphere and topographically-correlated signal.** If you don't anchor the prior on the tropospheric scale, the framework can absorb real elevation-correlated deformation into the troposphere state. Cross-check with non-deforming control regions.
- **Identifiability between ionosphere and long-wavelength orbital ramps.** Both are large-spatial-scale signals. Use the split-spectrum NISAR ionospheric correction as a strong prior; don't let the framework re-estimate ionosphere from scratch.
- **Phase wrapping at sensor level.** The framework needs unwrapped phase as input. Wrapping is not differentiable, so the LUEP handles wrapping ambiguities via discrete inference rather than continuous.
- **Memory blow-up at the outer loop.** Naive MCMC over high-dimensional hyperparameter space + millions of pixel-wise Kalman runs is expensive. Use the per-pixel parallelism aggressively and stagger pixel batches across MCMC samples.

### 1.6 Validation Plan

- **Synthetic test 1:** Single-pixel time series with known truth. Verify posterior coverage at 95% credible intervals.
- **Synthetic test 2:** Multi-pixel synthetic scene from the simulator (§3.3) with known displacement field + injected noise sources. Verify recovery of velocity field, strain rate field, and spatial uncertainty structure.
- **Simulation-based calibration:** Draw $\boldsymbol{\theta}$ from the prior, simulate data, run inference, and check that the rank of the true $\boldsymbol{\theta}$ among posterior draws is uniform over many replications. Non-uniform ranks are a proof of a bug in the model, likelihood, or sampler. This is both a standing regression test and the evidence a reviewer will want for the calibrated-uncertainty claims. See Talts et al. (2018).
- **Real-data test 1:** Southern California, existing Sentinel-1 data with dense GNSS. The quantitative baseline is our own least-squares estimator (§1.4), compared on identical inputs so that differences are attributable to the inference method rather than to preprocessing. GNSS is the independent ground truth for both — a comparison against GNSS is a stronger claim than agreement with another estimator, and should be framed that way in the manuscript. Published velocity fields for the region serve as an external sanity check.
- **Real-data test 2:** Recover known Hispaniola GNSS-based velocity field using InSAR + GNSS in a leave-one-out cross-validation.
- **Identifiability control:** A synthetic scene containing elevation-correlated *deformation* and zero troposphere. The framework must not absorb it into the tropospheric state. Run as a build-failing test, not as an occasional figure (see §1.5).

### 1.7 Key References (HBSS)

1. Segall, P., & Matthews, M. (1997). Time-dependent inversion of geodetic data. *JGR: Solid Earth*, 102, 22391–22409. https://doi.org/10.1029/97JB01795 — **Foundational NIF reference.**
2. Fukuda, J., Miyazaki, S., Higuchi, T., & Kato, T. (2008). Geodetic inversion for space–time distribution of fault slip with time-varying smoothing regularization. *Geophysical Journal International*, 173, 25–48. — **Closest precedent for hierarchical Bayesian state-space with time-varying hyperparameters.**
3. Bekaert, D. P. S., Segall, P., Wright, T. J., & Hooper, A. J. (2016). A network inversion filter combining GNSS and InSAR for tectonic slip modeling. *JGR: Solid Earth*, 121, 2069–2086. https://doi.org/10.1002/2015JB012638 — **NIF extended to InSAR; closest precedent for multi-source state-space.**
4. Dalaison, M., & Jolivet, R. (2020). A Kalman filter time series analysis method for InSAR. *JGR: Solid Earth*, 125, e2019JB019150. https://doi.org/10.1029/2019JB019150 — **KFTS: sequential single-sensor Kalman for InSAR. Direct architectural precedent for our inner loop.**
5. Sun, X., et al. (2022). Deriving 3-D surface deformation time series with strain model and Kalman filter from GNSS and InSAR data. *Remote Sensing*, 14, 2816. — **Recent demonstration of GNSS+InSAR Kalman fusion in Los Angeles.**
6. Maurer, J., Johnson, K., & Segall, P. (2017/2018). Bounding the moment deficit rate on crustal faults using geodetic data. *JGR: Solid Earth*, 122/123. — **Rigorous UQ methods for hazard-relevant products; relevant for downstream products.**
7. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. — **Standard textbook reference.**
8. Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554. — **NumPyro reference.**
9. Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs. — **JAX reference for the inner-loop implementation.**
10. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018). Validating Bayesian inference algorithms with simulation-based calibration. arXiv:1804.06788. — **The method for proving the inference is correct rather than merely plausible. Read in Week 2.**
11. Berardino, P., Fornaro, G., Lanari, R., & Sansosti, E. (2002). A new algorithm for surface deformation monitoring based on small baseline differential SAR interferograms. *IEEE TGRS*, 40(11), 2375–2383. — **SBAS; the network-inversion formulation underlying `hbss/baseline/`.**
12. Schmidt, D. A., & Bürgmann, R. (2003). Time-dependent land uplift and subsidence in the Santa Clara valley from a large interferogram set. *JGR: Solid Earth*, 108, 2416. — **Least-squares InSAR time-series estimation with rank-deficiency handling; the direct model for the baseline estimator.**
Jun'ichi Fukuda, Kaj M. Johnson, Mixed linear—non-linear inversion of crustal deformation data: Bayesian inference of model, weighting and regularization parameters, Geophysical Journal International, Volume 181, Issue 3, June 2010, Pages 1441–1458, https://doi.org/10.1111/j.1365-246X.2010.04564.x


---

## 2. Learned Noise Covariance Model (LNCM)

### 2.1 Summary

Train a neural network to predict the observation noise covariance $\mathbf{R}_t^{(s)}$ at each pixel and each acquisition, given a set of physically meaningful covariates: coherence $\gamma$, topographic slope and aspect, perpendicular baseline $B_\perp$, weather-model troposphere estimate, land cover, and time-of-year. The output replaces the standard $R \propto 1/\gamma^2$ parameterization with a learned relationship that adapts to local conditions.

The trained model is consumed by the HBSS inner loop as a per-pixel observation covariance. Training is two-stage: pretrain on Midwest US data (dense GNSS, corner reflectors, abundant samples) and fine-tune on Caribbean GNSS (target domain).

### 2.2 Architecture

**Backbone:** Convolutional encoder operating on local patches (32×32 pixels around each target pixel) of: coherence, topography (slope, aspect, elevation), land cover (categorical, one-hot encoded), and perpendicular baseline (broadcast as a scalar input). Concatenate per-pixel covariates (NWP troposphere, time-of-year) at the bottleneck.

**Decoder head:** Outputs the diagonal of $\mathbf{R}_t^{(s)}$ at the center pixel (i.e., per-pixel variance). For now, assume diagonal $\mathbf{R}$; off-diagonal terms can be added later if validation suggests they're needed.

**Probabilistic head:** Use deep ensembles (5 networks with different random initializations) or MC Dropout to produce a distribution over the predicted variance, not just a point estimate. This is important because the HBSS framework needs to propagate uncertainty about the noise model into the final posterior.

```
Input patch (32×32×N_features) ──→ CNN encoder ──→ bottleneck (128-d)
                                                          │
Per-pixel scalars (baseline, NWP, season) ────────────────┤
                                                          ↓
                                              MLP head ──→ predicted σ²(x)
```

Output is $\sigma^2_{\text{InSAR}}(\mathbf{x})$ in radians² (squared phase noise at the central pixel).

### 2.3 Training

**Loss function:** Gaussian negative log-likelihood (NLL) against residuals at ground-truth control points:

$$
\mathcal{L} = \sum_i \left[ \frac{r_i^2}{2\sigma_i^2} + \frac{1}{2}\log\sigma_i^2 \right] + \lambda \cdot \mathcal{L}_{\text{cal}}
$$

where $r_i$ is the residual between the InSAR phase and the GNSS/corner-reflector-derived ground truth at control point $i$, $\sigma_i^2$ is the predicted variance, and $\mathcal{L}_{\text{cal}}$ is an expected-calibration-error regularizer to encourage probability calibration.

**Training data construction:**

*Stage 1 (Midwest pretraining):* Use existing Sentinel-1 and corner-reflector observations from the Midwest US over a 3–5 year window. Compute InSAR phase at each corner-reflector pixel and at GNSS-station-collocated pixels. Compute ground-truth residual as (InSAR phase) − (predicted phase from GNSS time series). Build a training set with ~10⁴–10⁵ training pairs across the network.

*Stage 2 (Caribbean fine-tuning):* Use existing Sentinel-1 + GNSS data in the Caribbean (especially Puerto Rico and the Dominican Republic where GNSS is denser). Smaller training set (~10³ pairs likely) but in-domain. Fine-tune all layers, or freeze encoder and fine-tune only the head depending on overfitting risk.

*Stage 3 (NISAR-specific fine-tuning):* Once NISAR data accumulates (Year 2), repeat fine-tuning with NISAR phase observations to capture the L-band-specific noise structure. The framework should be retrainable per-sensor.

**Optimization:** Adam with cosine learning rate schedule. Early stopping on held-out validation. Batch size 64–256 patches. Likely converges in 50–200 epochs depending on data volume.

### 2.4 Validation

- **In-distribution test:** Held-out GNSS stations in the same region as training. Verify that 95% credible intervals from the predicted $\sigma^2$ cover ~95% of residuals.
- **Out-of-distribution test:** Apply to a different region (e.g., trained on Midwest, evaluated on Hispaniola) without fine-tuning. Document the degradation.
- **Comparison baseline:** Compare against constant-$R$ and $R \propto 1/\gamma^2$ baselines on the same held-out sets. The learned model should outperform at least one of these.
- **Downstream test:** Plug the LNCM into the HBSS framework and verify that velocity posterior coverage improves over the parametric baselines.

### 2.5 Pitfalls

- **Distribution shift between Midwest and Caribbean.** Midwest is largely agricultural with strong temporal decorrelation but uniform topography. Caribbean is mountainous, vegetated, with different decorrelation drivers. Fine-tuning will not fully bridge this — validation must explicitly test cross-region generalization.
- **Calibration vs. accuracy tradeoff.** Networks trained on NLL can produce sharp predictions that are miscalibrated. Always include a calibration regularizer or use temperature scaling at inference.
- **Lack of corner reflectors in target region.** Caribbean has very few permanent corner reflectors. Fall back to GNSS-collocated InSAR pixels as ground truth, accepting that the spatial sampling is sparser.
- **Training-pair construction is the bottleneck, not the network.** Assembling $10^4$–$10^5$ labeled residuals with defensible spatial and temporal blocking is the bulk of the work; the CNN itself is roughly a week. Schedule accordingly, and note that InSAR pixels near the same GNSS station on adjacent dates are near-duplicates — an unblocked random train/test split will report excellent held-out performance that is entirely fictitious.

### 2.6 Integration with HBSS

The LNCM is applied **offline**, not called from inside the sampler. After training, run the model over the tile store and write the predicted variance fields into Zarr alongside the observations; the deep-ensemble spread (§2.2) is written as additional bands and consumed by the HBSS as a hyperprior on the noise scale rather than as a point estimate. The JAX inner loop then reads an array and never touches torch.

This is not merely convenient. It keeps the `hbss` package free of a torch dependency (which is what makes the "Year 1 completes without ML" fallback real rather than aspirational), it removes the problem of JAX and torch contending for the same GPU inside one process, and it makes every inference run reproducible from a versioned, checksummed model artifact. A live in-process model can be added later behind the same `NoiseModel` interface if a use case ever requires it.

Every exported artifact carries a model card: training-data manifest hash, code SHA, config, and validation metrics. Calibration is a **gate** on export, not a finding reported afterward — a model whose held-out expected calibration error exceeds threshold does not get written.

### 2.7 Key References (LNCM)

1. Hanssen, R. F. (2001). *Radar Interferometry: Data Interpretation and Error Analysis*. Springer. — **Standard reference for InSAR noise models, baselines, decorrelation theory.**
2. Zebker, H. A., & Villasenor, J. (1992). Decorrelation in interferometric radar echoes. *IEEE TGRS*, 30(5), 950–959. — **Theoretical basis for coherence-decorrelation relationships.**
3. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS 2017*. arXiv:1703.04977 — **Foundational reference for Bayesian deep learning approaches to uncertainty.**
4. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML 2016*. — **MC Dropout reference.**
5. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS 2017*. — **Deep ensembles reference.**
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*. — **Calibration / temperature scaling.**
7. Rouet-Leduc, B., Jolivet, R., Dalaison, M., Johnson, P. A., & Hulbert, C. (2021). Autonomous extraction of millimeter-scale deformation in InSAR time series using deep learning. *Nature Communications*, 12, 6480. — **Closest precedent for ML noise modeling in InSAR.**
8. Cao, Y., Jónsson, S., & Li, Z. (2021). Advanced InSAR tropospheric corrections from global atmospheric models that incorporate spatial stochastic properties of the troposphere. *JGR: Solid Earth*, 126, e2020JB020952. — **State-of-the-art parametric noise modeling baseline to beat.**
9. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE TKDE*, 22(10), 1345–1359. — **Transfer learning theory background for the two-stage training paradigm.**

---

## 3. Learned Unwrapping-Error Proposal (LUEP)

### 3.1 Summary

Train a probabilistic U-Net to take an interferogram (wrapped phase + coherence + connected-component map + topography) as input and output, per connected component, a probability distribution over candidate unwrapping-error corrections (integer multiples of $2\pi$). This distribution serves as a *learned proposal* in the HBSS outer-loop MCMC sampling of discrete unwrapping-error states $\mathbf{k}$.

Critically, the network does *not* decide the unwrapping correction; it proposes plausible candidates, which the MCMC accepts or rejects based on the full posterior. This preserves the calibrated uncertainty chain through the framework.

### 3.2 Architecture

**Backbone:** U-Net with 4–5 down/up sampling levels. Input channels: wrapped phase (sin and cos components, 2 channels), coherence, connected-components-ID (categorical, one-hot or embedded), topography (elevation + slope).

**Connected-component reasoning:** Add a graph-based message-passing layer at the bottleneck of the U-Net. Nodes = connected components; node features = aggregated U-Net features over each component; edges = component-adjacency relations. Two rounds of message passing capture the "unwrapping errors propagate within and between adjacent components" structure.

**Output head:** Per-connected-component distribution over candidate corrections $\{-2\pi, 0, +2\pi, \pm 4\pi, \ldots\}$. In practice, restrict to the most common values (typically $\{-2\pi, 0, +2\pi\}$, occasionally larger multiples). Use softmax over discrete candidates with calibration regularization.

```
Wrapped phase + coherence + CC map + topography (H×W×C)
                        ↓
              U-Net encoder (downsampling)
                        ↓
        Bottleneck features pooled per CC
                        ↓
            GNN message passing (2 rounds)
                        ↓
       Per-CC softmax over {-2π, 0, +2π, ...}
```

### 3.3 Training Data

**The hardest part of this component, and therefore the part scheduled earliest.** The simulator that produces source 1 below lives in `hbss/simulate/` and is built in two tiers, starting in Year 1:

- **`v0` (Year 1 Q3, postdoc/PI, ~2 weeks).** Known displacement field, spatially correlated Gaussian noise, injected $2\pi$ errors. Makes no claim to physical realism. Sufficient for §1.6 synthetic test 2 and for CI integration tests. Exists behind the interface the `v1` implementation will later satisfy.
- **`v1` (Year 1 Q4 onward, PhD student).** Physically-based decorrelation, DEM-driven topography, unwrapper-in-the-loop labeling. This is the student's first-paper deliverable; see `HBSS-dev-plan.md` §5.

The `v0` tier exists specifically so that HBSS validation never waits on the student's availability. This is a general policy for the project, not a one-off arrangement.

Three data sources, combined:

*1. Physics-simulated synthetic.* Use a coherence/decorrelation simulator (based on Zebker & Villasenor 1992 + thermal and volumetric scattering) to generate realistic noisy interferograms with known ground truth. Simulate the unwrapping process (e.g., SNAPHU) and compare to the true unwrapped phase to label which pixels/components have errors. Generate ~10⁴–10⁵ synthetic scenes spanning diverse topography and decorrelation regimes. Note that the labeled benchmark this produces — unwrapping-error rates as a function of coherence, topography, and baseline — is a publishable community resource in its own right, independent of whether the LUEP itself succeeds.

*2. Real interferograms with high-coherence reference areas.* In high-coherence sub-regions of real interferograms, unwrapping errors are extremely rare and can be assumed absent. Pair these with low-coherence sub-regions where errors occur. Use the high-coherence sub-region as a constraint to identify which low-coherence pixels have errors. Yields ~10³ real labeled pairs.

*3. Corner-reflector and GNSS-anchored real interferograms.* In Midwest US data, use the dense corner-reflector network to anchor absolute phase across multiple regions of an interferogram; deviations from this anchor identify unwrapping errors. Smaller sample but high-fidelity labels.

Two-stage training as with LNCM: pretrain on simulated, fine-tune on real labeled data.

### 3.4 Inference and HBSS Integration

At each MCMC step in the outer loop, for each connected component $c$ with current state $k_c$:

1. Query the LUEP for $p(k_c \mid \text{interferogram features})$.
2. Sample a candidate $k_c' \sim q(k_c' \mid k_c, \text{LUEP output})$ where $q$ mixes LUEP and a uniform proposal for robustness.
3. Compute Metropolis-Hastings acceptance ratio using full posterior + LUEP-based proposal correction.
4. Accept or reject.

The LUEP can be trained offline (before the HBSS run); during HBSS inference, it is a fixed proposal distribution. This separates the heavy ML training from the Bayesian inference.

Because the proposal is fixed, step 1 above should be **precomputed rather than evaluated live**: for each interferogram, export a table of per-connected-component candidate probabilities, which the sampler reads. As with the LNCM (§2.6), this keeps torch out of the `hbss` package and out of the MCMC loop entirely.

**Interface warning.** The `UnwrapProposal` interface returns $\log q(k \mid k') - \log q(k' \mid k)$ from the very first implementation, even though that quantity is identically zero for the symmetric uniform proposal used in Year 1. Retrofitting the proposal ratio when the LUEP arrives in Year 3 is precisely where a silent Metropolis–Hastings bug gets introduced, and such a bug produces plausible-looking, wrong posteriors that no amount of eyeballing will catch.

### 3.5 Validation

- **In-distribution test:** Held-out synthetic interferograms with known ground truth. Verify that the top-1 prediction is correct on >X% of components (target depends on noise regime; quantify in validation).
- **Calibration test:** Verify that the predicted probabilities are well-calibrated (e.g., for components where the LUEP says "70% chance of $+2\pi$", verify the empirical rate is ~70%).
- **Integration test:** Compare HBSS MCMC acceptance rates and convergence speed with LUEP proposals vs. uniform proposals. Target: 2–10× speedup in chain mixing.
- **Cross-region test:** Train on Midwest synthetic, evaluate on Caribbean real data. Document the degradation.

### 3.6 Pitfalls

- **Synthetic-to-real gap.** Simulated decorrelation is a noisy approximation to real decorrelation. Without sufficient real-data fine-tuning, the model will look great on synthetic and underperform on real interferograms. Budget time for real-data labeling.
- **Connected-component identification.** The framework assumes you have a stable connected-component map. This is upstream of the LUEP; verify your unwrapping pipeline (SNAPHU or Dolphin) produces reproducible CC maps.
- **Combinatorial explosion in MCMC.** Even with the LUEP, sampling discrete states over $\sim$10³–10⁴ connected components is expensive. Block-update components by spatial proximity to improve mixing.
- **Fallback if LUEP fails.** If LUEP doesn't converge to calibrated predictions during training, the HBSS framework can fall back to uniform proposals — slower but well-posed. This is the documented Year-1-completes-without-AI/ML risk-management posture.

### 3.7 Key References (LUEP)

1. Sica, F., Calvanese, F., Scarpa, G., & Rizzoli, P. (2020). A CNN-based coherence-driven approach for InSAR phase unwrapping. *IEEE GRSL*, 17(6), 1023–1027. — **PhiNet: CNN-based phase unwrapping.**
2. Spoorthi, G. E., Gorthi, S., & Gorthi, R. K. S. S. (2020). PhaseNet: A deep convolutional neural network for two-dimensional phase unwrapping. *IEEE SPL*. — **PhaseNet: foundational deep learning for unwrapping.**
3. Yu, H., et al. (2023). PUNet: A deep learning method for phase unwrapping. *Remote Sensing*. — **Recent SOTA in deep learning for phase unwrapping.**
4. Chen, C. W., & Zebker, H. A. (2002). Phase unwrapping for large SAR interferograms: statistical segmentation and generalized network models. *IEEE TGRS*. — **SNAPHU reference (the standard unwrapper).**
5. Ansari, H., De Zan, F., & Bamler, R. (2017). Sequential estimator: toward efficient InSAR time series analysis. *IEEE TGRS*. — **Phase-linking foundation; relevant context for unwrapping in time series.**
6. Mirzaee, S., Amelung, F., & Fattahi, H. (2023). Non-linear phase linking using joined distributed and persistent scatterers (MiaplPy). *Computers & Geosciences*. — **Modern phase-linking and unwrapping pipeline.**
7. Andrieu, C., De Freitas, N., Doucet, A., & Jordan, M. I. (2003). An introduction to MCMC for machine learning. *Machine Learning*, 50(1-2), 5–43. — **Foundational MCMC; relevant to the proposal-distribution framework.**
8. Levy, D., Hoffman, M. D., & Sohl-Dickstein, J. (2018). Generalizing Hamiltonian Monte Carlo with neural networks. *ICLR 2018*. — **Learned proposal distributions in MCMC; conceptual precedent for the LUEP-as-proposal architecture.**
9. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. *MICCAI 2015*. — **U-Net backbone reference.**
10. Battaglia, P. W., et al. (2018). Relational inductive biases, deep learning, and graph networks. arXiv:1806.01261. — **GNN reference for connected-component reasoning.**

---

## 4. Cross-Cutting Implementation Concerns

### 4.1 Data Pipeline

- All InSAR data ingested via Dolphin or ISCE3 outputs (RSLCs and GUNWs).
- All GNSS data ingested via GAGE Facility / EarthScope archive formats (typically .pos files).
- Weather model fields via RAiDER ZTD output (NetCDF).
- Time-series estimation and the associated corrections are implemented in-project (`hbss/baseline/`, §1.4) rather than delegated to an external package. The problem is small enough that owning it is cheaper than depending on it, and it keeps the dependency surface and the license situation clean.
- Standardize all internal representations to xarray DataArrays with CF-compliant metadata. This makes Zarr-on-disk storage trivial and downstream analysis cleaner.
- Data never lives in the repository. A `registry.yaml` maps logical dataset names to URI, checksum, size, and provenance; a fetch command populates `$HBSS_DATA_ROOT`. Test fixtures are ≤ 5 MB and generated by `hbss/simulate/` wherever possible.

### 4.2 Environment

- Python ≥ 3.11.
- Core: NumPy, SciPy, xarray, pandas, h5py, zarr.
- ML stack: PyTorch (for LNCM and LUEP training), JAX + Flax (for the HBSS inner loop), NumPyro (for the HBSS outer loop). Pick PyTorch over JAX for the ML components because the ecosystem is richer; pick JAX for the HBSS because of NumPyro integration and JIT compilation for the per-pixel inner loop.
- InSAR-specific: ISCE3, Dolphin, RAiDER. Install via conda-forge.
- **Two environments, not one.** `core` (JAX/NumPyro) and `learned` (torch). Do not merge them; co-installing CUDA-enabled JAX and CUDA-enabled torch is a recurring time sink, and the offline-export integration design (§2.6, §3.4) means there is no reason to. The conda-forge base plus PyPI layer is handled by `pixi` (one lockfile spanning both), with `environment.yml` + `uv` as the fallback if that causes friction. Decide in week 1 and record the decision.
- Optional GPU: CUDA 12.x. Make GPU use optional — the framework should still run on CPU at reduced speed.

### 4.3 Testing

Full strategy in `HBSS-dev-plan.md` §2. In brief, tests are tiered by runtime — unit (<1 min, every push), statistical (~5 min, every PR), integration (~20 min, nightly), benchmark and real-data (hours, weekly). Two techniques carry most of the correctness load, because Bayesian inference code fails silently by producing smooth, plausible, wrong posteriors that ordinary unit tests will not catch:

- **Dense-solve reference** for the Kalman/RTS inner loop (§1.4), matched to ~$10^{-9}$, including the marginal likelihood.
- **Simulation-based calibration** on the full posterior (§1.6), at the single-pixel level per PR and the tile level nightly.

Supporting these: hypothesis for the linear algebra (covariances stay symmetric positive-definite; information and covariance filter forms agree; a zero-noise system reproduces its input exactly); a Geweke joint-distribution test for the Metropolis-within-Gibbs sweep over $\mathbf{k}$; integration tests on 10×10-pixel, 10-epoch synthetic scenes asserting 95% interval coverage in [0.90, 0.98] and $\hat{R} < 1.01$; and performance regression on a 256×256-pixel, 30-epoch tile.

For the learned components: overfit-one-batch as a smoke test, a leakage guard asserting spatial and temporal blocking in the train/val/test splits, and calibration thresholds as pass/fail gates on artifact export.

**Provenance.** Every run writes a record containing git SHA and dirty flag, resolved config, input data manifest hashes, RNG seeds, package versions, host/GPU, and wall time. Built in Year 1 Q1, before any science happens — retrofitting provenance after eighteen months of results never quite works.

### 4.4 Documentation

- Sphinx + autodoc + ReadTheDocs.
- A tutorial notebook for each major workflow: ingest NISAR data, build the model, run inference, post-process to velocity field. Build these in `examples/` and have them auto-execute in CI.
- API reference for each public module.

### 4.5 Release Plan

- **v0.1 (Year 1, Q4):** HBSS framework with parametric noise models and naive discrete MCMC. Full I/O, least-squares baseline estimator, simulator v0, validation on synthetic and southern California data, basic documentation. **This release is the project's floor** — it is what makes the framework defensible if either learned component fails, and it ships before either is attempted.
- **v0.5 (Year 2, Q3):** LNCM integrated; validation on real Hispaniola/Guatemala data.
- **v1.0 (Year 2, Q4):** Feature-complete release. Submit accompanying methods paper.
- **v1.5 (Year 3, Q2):** LUEP integrated; full plate-boundary results.
- **v2.0 (Year 3, Q4):** Final release with full documentation, tutorial expansion based on workshop feedback, and the regional Caribbean dataset.

---

## 5. Consolidated Reading Priority

For someone picking up the project cold, read in this order:

**Week 1 — Foundations:**
1. Segall & Matthews (1997) — understand the NIF.
2. Fukuda et al. (2008) — understand hierarchical Bayesian state-space inversion.
3. Bekaert et al. (2016) — understand multi-source NIF.
4. Dalaison & Jolivet (2020) — understand KFTS architecture.

**Week 2 — Methods:**
5. Hanssen (2001), Chapters 4 and 6 — InSAR noise theory.
6. Cao et al. (2021) — modern tropospheric correction.
7. Gelman et al., *Bayesian Data Analysis*, Chapters 5–6 — hierarchical Bayes refresher.
8. Talts et al. (2018) — simulation-based calibration. Short, and it defines how the project will demonstrate that its inference is correct.
9. Berardino et al. (2002) and Schmidt & Bürgmann (2003) — the least-squares network inversion underlying `hbss/baseline/`.

**Week 3 — ML components:**
10. Kendall & Gal (2017) — Bayesian deep learning.
11. Lakshminarayanan et al. (2017) — deep ensembles.
12. Rouet-Leduc et al. (2021) — closest InSAR ML precedent.
13. Sica et al. (2020), Yu et al. (2023) — phase unwrapping with deep learning.

**Week 4 — Implementation:**
14. NumPyro tutorials (https://num.pyro.ai/).
15. JAX documentation, particularly `vmap` and `scan`.
16. Dolphin and ISCE3 user documentation.

---

## 6. Kickoff Decisions

These were deferred from the proposal stage. All six are now resolved; each gets a one-page architecture decision record in `docs/decisions/` during Year 1 Q1, to be confirmed or revisited with the postdoc.

1. **JAX-based custom Kalman**, with the dense-matrix reference implementation kept permanently in the test suite (§1.4) rather than treated as a prototyping fallback. The reference is the debugging instrument, and it is what makes the fast path trustworthy.
2. **Diagonal $R$** initially — but `NoiseModel` returns a covariance sum type (`Diagonal | Dense`) from day one, so adding off-diagonal support later is an additive change rather than a breaking one.
3. **Physics-simulated decorrelation.** Settled by dual use: the same simulator is required for HBSS synthetic validation regardless of what the LUEP ends up needing (§3.3).
4. **Monorepo**, two installable packages (`hbss`, `hbss-learned`), one-way dependency enforced by a CI job that installs `hbss` alone into a torch-free environment. See §0 and `HBSS-dev-plan.md` §1.
5. **Apache 2.0**, with SPDX headers and a `NOTICE` file. Confirm with OSP in week 1 and verify the licenses of ISCE3, Dolphin, and RAiDER at the same time — all are expected to be permissive. The decision to implement time-series estimation in-project (§1.4) removes the only anticipated copyleft exposure.
6. **GitHub Actions** for unit, statistical, and integration tiers. One self-hosted GPU runner for nightly and weekly benchmarks only; never on the pull-request path, so that a dead lab machine cannot block a merge.

Remaining open questions, to be resolved by evidence rather than at kickoff:

- Does NUTS over $\boldsymbol{\theta}$ compose cleanly with the JAX Kalman marginal likelihood and Gibbs over $\mathbf{k}$ in NumPyro? Answered by the Year 1 Q1 spike; plan B is a hand-rolled Gibbs outer loop or empirical Bayes on the marginal likelihood.
- Is full-scene MCMC affordable at the scale of the plate-boundary application? Answered by the Year 1 Q2 compute budget model, before spatial coupling is built on top.

---

*End of HBSS Implementation Plan. See `HBSS-dev-plan.md` for the software development and testing plan.*
