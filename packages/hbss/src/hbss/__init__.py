"""HBSS -- hierarchical Bayesian state-space estimation for multi-sensor InSAR.

This package holds the inference engine, I/O, the least-squares baseline
estimator, the synthetic scene simulator, and the validation harness. It
depends on JAX and NumPyro.

It must never import torch, directly or transitively. The learned components
(LNCM, LUEP) live in the separate ``hbss-learned`` package and reach this one
only by writing artifacts that it reads: predicted noise fields in the tile
store, candidate probability tables for the sampler. A CI job installs this
package alone into a torch-free environment and runs the full test suite, so a
violation fails the build rather than being caught in review.

See HBSS-dev-plan.md section 1.3.
"""

__version__ = "0.0.0.dev0"
