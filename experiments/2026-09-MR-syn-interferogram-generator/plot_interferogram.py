#!/usr/bin/env python3

"""
Script to plot a synthetic interferogram created by create_synthetic_interferogram.py. Shows the ground truth
(uncorrupted phase) beside the decorrelated (noisy) phase. 

Usage: 
    python plot_interferogram.py <path/to/interferogram_XXXX.npz>

Saves a .png with the same name next to the .npz file and opens the figure. 
"""

import json 
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_interferogram(npz_path):
    '''Draws the clean and noisy wrapped phase side by side and saves a PNG.'''
    data = np.load(npz_path)
    meta = json.loads(str(data["metadata"]))
 
    # The grid runs from -extent to +extent km in both directions
    extent_km = meta["grid_extent_km"]
    extent = [-extent_km, extent_km, -extent_km, extent_km]
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    panels = [
        (data["phase_wrapped"], "Uncorrupted"),
        (data["phase_noisy"], "With noise and decorrelation"),
    ]
    for ax, (phase, title) in zip(axes, panels):
        image = ax.imshow(phase, origin="lower", extent=extent,
                          cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(title)
        ax.set_xlabel("East (km)")
    axes[0].set_ylabel("North (km)")
    fig.colorbar(image, ax=axes, shrink=0.85, label="Wrapped phase (rad)")
 
    fig.suptitle(
        f"{os.path.basename(npz_path)}   "
        f"Mw {meta['Mw']:.1f}, strike {meta['strike_deg']:.0f}°, "
        f"dip {meta['dip_deg']:.0f}°, depth {meta['depth_km']:.1f} km, "
        f"mean coherence {data['coherence'].mean():.2f}"
    )
 
    png_path = os.path.splitext(npz_path)[0] + ".png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved figure to: {png_path}")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Error: You must specify the .npz file to plot.")
        print("Usage: python plot_interferogram.py <path/to/interferogram_XXXX.npz>")
        sys.exit(1)
 
    npz_path = sys.argv[1]
    if not os.path.isfile(npz_path):
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)
 
    plot_interferogram(npz_path)

if __name__ == "__main__":
    main()

