"""
Generate synthetic interferograms with explicitly set decorrelation noise.

Run from the eq-insar repo root:
    python decorrelation_examples.py
Saves decorrelation_examples.png and decorrelation_examples.npz.
"""

import numpy as np
import matplotlib.pyplot as plt

from eq_insar import generate_synthetic_insar

# Shared earthquake + grid settings for every case
SOURCE = dict(
    Mw=4.0, strike_deg=125, dip_deg=52, rake_deg=180, depth_km=5,
    satellite="sentinel1", grid_size=256, grid_extent_km=50, #Grid is 256 by 256 pixels. 
    add_noise=False,   # isolate decorrelation; set True to add atmosphere too
    seed=42,
)

# ---------------------------------------------------------------------------
# 1. Explicit parameters for each decorrelation source
# ---------------------------------------------------------------------------
cases = {
    "No decorrelation": dict(
        background_coherence=1.0, # Coherence set to 1.0 to create no decorrelation.
    ),
    "Background only (γ = 0.6)": dict(
        background_coherence=0.6, # Coherence reduced to 60% distributed evenly across the interferogram.
        n_looks=4, # Each pixel treated as an avg. of 4 independent measurements. More looks, less noise w/i same coherence. 
    ),
    "Vegetation, 48-day pair": dict(
        background_coherence=0.9, # Coherence reduced to 90% distributed evenly across the interferogram.
        vegetation_fraction=0.4, # 40% of the grid covered with vegetation patches. 
        # Temporal model: y = y-inf + (1 - y-inf)*e^(-delta(t)/tau), where delta(t) = 48, tau = 20, y-inf = 0.1
        # y is the coherence. The smaller the y the smaller the coherence. In this example, y = 0.18.
        temporal_baseline_days=48, 
        vegetation_tau_days=20,
        vegetation_coherence_long_term=0.1,
    ),
    "Snow patches": dict(
        background_coherence=0.9,
        snow_fraction=0.25, # 25% of grid covered with snow patches.
        snow_coherence=0.05, # Coherence in the regions of snow set to 5%. 
    ),
    "Surface rupture (40 km)": dict(
        background_coherence=0.9,
        rupture_length_km=40, # Rupture feature 40 km in length.
        rupture_width_km=4, # Rupture feature 4 km in width.
        rupture_coherence=0.0, # Coherence of rupture area set to 0%.
        # strike/center default to the fault's strike_deg and xcen/ycen_km
    ),
    "All combined": dict(
        background_coherence=0.85,
        vegetation_fraction=0.3,
        temporal_baseline_days=24,
        snow_fraction=0.1,
        rupture_length_km="auto",   # Wells & Coppersmith length for Mw
        rupture_width_km=3,
    ),
}

# ---------------------------------------------------------------------------
# 2. Your own masks: put decorrelation exactly where you want it
# ---------------------------------------------------------------------------
n = SOURCE["grid_size"]
x = np.linspace(-SOURCE["grid_extent_km"], SOURCE["grid_extent_km"], n)
X, Y = np.meshgrid(x, x)                 # km; row 0 = south (origin='lower')

snow_mask = (Y > 20).astype(float)        # snow over the northern 30 km
veg_mask = np.clip(1 - np.hypot(X + 25, Y + 25) / 15, 0, 1)  # forest disc, SW

cases["Custom masks (snow N, forest SW)"] = dict(
    background_coherence=0.9,
    snow_mask=snow_mask, snow_coherence=0.05,
    vegetation_mask=veg_mask, temporal_baseline_days=60,
)

# ---------------------------------------------------------------------------
# Generate, plot, save
# ---------------------------------------------------------------------------
# The results here are the output that can be fed into ML pipeline.
results = {}
for name, params in cases.items():
    results[name] = generate_synthetic_insar(
        **SOURCE, add_decorrelation=True, decorrelation_params=params, #Pass SOURCE metrics w/o retyping them.
    )

# Create figure
fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 6.6))
extent = [x[0], x[-1], x[0], x[-1]]
for col, (name, res) in enumerate(results.items()):
    ax = axes[0, col]
    im_p = ax.imshow(res["phase_noisy"], origin="lower", extent=extent, #origin='lower' for north at top of grid.
                     cmap="twilight", vmin=-np.pi, vmax=np.pi) #cmap='twilight' is cyclic color map for phase.
    ax.set_title(name, fontsize=9)
    ax = axes[1, col]
    im_c = ax.imshow(res["coherence"], origin="lower", extent=extent,
                     cmap="gray", vmin=0, vmax=1)
    for a in axes[:, col]:
        a.set_xticks([]); a.set_yticks([])
axes[0, 0].set_ylabel("Wrapped phase")
axes[1, 0].set_ylabel("Coherence")
fig.colorbar(im_p, ax=axes[0, :], shrink=0.8, label="rad")
fig.colorbar(im_c, ax=axes[1, :], shrink=0.8, label="γ")
fig.savefig("decorrelation_examples_2.png", dpi=120, bbox_inches="tight") #TODO: Append timestamp to pic file name.

np.savez_compressed(
    "decorrelation_examples_2.npz",
    names=np.array(list(results)),
    phase=np.stack([r["phase_noisy"] for r in results.values()]),
    coherence=np.stack([r["coherence"] for r in results.values()]),
)
print("Saved decorrelation_examples.png and decorrelation_examples.npz")


# Create generator object to streamline the workflow. 
# Experient with unwrapping error generation. 
# Account for water bodies, like reivers and lakes. 
# For each connected component add a randomized unwrapping error. 
# Push your code or create pull request so this is on github. 

