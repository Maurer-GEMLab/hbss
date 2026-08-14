"""Synthetic scene generator.

Two tiers: v0 (correlated Gaussian noise, injected 2-pi errors -- enough for
validation and CI) and v1 (physically-based decorrelation, unwrapper in the
loop -- the LUEP's training-data supply). See HBSS-dev-plan.md section 5.1.
"""
