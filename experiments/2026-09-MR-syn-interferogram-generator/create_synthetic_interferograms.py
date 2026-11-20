#!/usr/bin/env python3

"""
Script to automatically generate synthetic interferograms as the first step in the ML pipeline. 

Run this script from a folder that contains the local eq-insar copy:
    python create_synthetic_interferograms.py <Number of Interferograms>

Saves one compressed .npz file per interferogram, as well as a manifest.csv listing every file and its 
source parameters to the synthetic_interferograms folder. 
"""

import csv 
from datetime import datetime
import json
import os
import sys

# Dynamically find where script is saved and point to local instead of remote eq-insar. 
script_dir = os.path.dirname(os.path.abspath(__file__))
local_repo_path = os.path.join(script_dir, "eq-insar", "src")
sys.path.insert(0, local_repo_path)

import numpy as np
from eq_insar import generate_synthetic_insar

# Constants
OUTPUT_DIR = os.path.join(script_dir, "synthetic_interferograms")
BASE_SEED = 42 # TODO: Will be dynamically set by the user.

def create_sample_params(rng):
    '''Creates randomized source and decorrelation parameters for a single interferogram.'''
    source = dict(
        Mw=rng.uniform(5.5, 7.0), 
        strike_deg=rng.uniform(0, 360), 
        dip_deg=rng.uniform(30, 90), 
        rake_deg=rng.uniform(-180, 180),
        depth_km=rng.uniform(2, 15),
        satellite="sentinel1", # TODO: Verify whether or not we should randomize satellite.
        grid_size=256,
        grid_extent_km=50,
        add_noise=True,
    )

    decorrelation = dict(
        background_coherence=rng.uniform(0.7, 0.95), 
        vegetation_fraction=rng.uniform(0.0, 0.5),
        temporal_baseline_days=float(rng.choice([6, 12, 24, 48])),
        snow_fraction=rng.uniform(0.0, 0.2),
        rupture_length_km="auto",
    )
    
    return source, decorrelation

def to_json_safe(obj):
    '''Converts NumPy types so metadata can be written as JSON.'''
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot convert {type(obj)} to JSON")

def save_interferogram(result, filepath):
    '''Saves one interferogram and its labels to a compressed .npz file.'''
    weights = result["decorrelation_weights"]
    np.savez_compressed(
        filepath,
        # Model input
        phase_noisy=result["phase_noisy"].astype(np.float32),
        # Possible targets / labels
        phase_wrapped=result["phase_wrapped"].astype(np.float32),
        los_displacement=result["los_displacement"].astype(np.float32),
        coherence=result["coherence"].astype(np.float32),
        vegetation_weight=weights["vegetation"].astype(np.float32),
        snow_weight=weights["snow"].astype(np.float32),
        rupture_weight=weights["rupture"].astype(np.float32),
        # Every parameter used, stored as a JSON string
        metadata=json.dumps(result["metadata"], default=to_json_safe),
    )


def main(): 

    ''' Validation '''
    if len(sys.argv) < 2:
        print("Error: You must specify the number of interferograms you would like to generate.")
        print("Usage: python create_synthetic_interferograms.py <your_number_here>")
        sys.exit(1)

    # Cast arg to int
    try:
        num_interferograms = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' Argument must be a natural number (Positive int).")
        sys.exit(1)

    # Check arg valid int
    if num_interferograms < 1: 
        print("Error: The number of interferograms must be at least 1."); 
        sys.exit(1)


    ''' Generation '''
    # Create subfolders based on time to prevent overwriting of older-created interferograms.
    subfolder = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True) # Create folder if not present, otherwise continue.
    
    rng = np.random.default_rng(BASE_SEED)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    with open(manifest_path, "w", newline="") as manifest_file:
        writer=None
        
        for i in range(num_interferograms):
            source, decorrelation = create_sample_params(rng)
            result = generate_synthetic_insar(
                **source,
                add_decorrelation=True,
                decorrelation_params=decorrelation,
                seed=BASE_SEED + i,
            )
 
            filename = f"interferogram_{i:05d}.npz"
            save_interferogram(result, os.path.join(output_dir, filename))

            # Store metadata for each interferogram.
            meta = result["metadata"]
            decorr = meta["decorrelation_params"]
            row = {
                "file": filename,
                "seed": BASE_SEED + i,
                "Mw": meta["Mw"],
                "strike_deg": meta["strike_deg"],
                "dip_deg": meta["dip_deg"],
                "rake_deg": meta["rake_deg"],
                "depth_km": meta["depth_km"],
                "background_coherence": decorr["background_coherence"],
                "vegetation_fraction": decorr["vegetation_fraction"],
                "temporal_baseline_days": decorr["temporal_baseline_days"],
                "snow_fraction": decorr["snow_fraction"],
                "rupture_length_km": decorr["rupture_length_km"],
                "mean_coherence": float(result["coherence"].mean()),
            }

            if writer is None:
                writer = csv.DictWriter(manifest_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)

            print(f"{i + 1} out of {num_interferograms} generated. Saved to: {filename}")
 
    print(f"Generation complete. Files are saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



