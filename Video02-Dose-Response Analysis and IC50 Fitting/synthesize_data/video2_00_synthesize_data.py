"""
Author: Sreenivas Bhattiprolu (DigitalSreeni)
YouTube: youtube.com/@DigitalSreeni
GitHub:  github.com/bnsreenu

AI for Drug Discovery -- Video 2
Synthetic Dose-Response Data Generator
=======================================
Generates realistic synthetic dose-response curves for tutorial use.

Each curve follows the 4-parameter logistic (4PL) model:

    response = Bottom + (Top - Bottom) / (1 + (IC50 / concentration)^Hill)

Five compound types are generated, each illustrating a scenario that
scientists encounter in real HTS or lead optimisation work:

    1. clean_sigmoid      -- ideal curve, easy to fit
    2. noisy_sigmoid      -- same curve with substantial measurement noise
    3. hook_effect        -- activity drops at high concentration (aggregation
                            or solubility artefact)
    4. incomplete_sigmoid -- highest concentrations not tested; curve has not
                            reached its lower plateau
    5. inactive           -- flat line, no meaningful activity

Output
------
Saves one CSV per compound to the specified output directory.
Each CSV has columns: compound_id, concentration_uM, response_pct

Usage
-----
    python video2_00_synthesize_data.py

Edit OUTPUT_DIR below to point at your Google Drive data folder before
uploading, or run directly on Colab after mounting Drive.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration -- edit OUTPUT_DIR to match your Drive path
# ---------------------------------------------------------------------------

OUTPUT_DIR = (
    "/content/drive/MyDrive/ColabNotebooks/"
    "AI_for_drug_discovery/Video2_dose_response_curves/data"
)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 4PL model
# ---------------------------------------------------------------------------

def four_pl(concentration, bottom, top, ic50, hill):
    """
    Four-parameter logistic (4PL) dose-response model.

    Parameters
    ----------
    concentration : array-like
        Compound concentrations (same units as ic50).
    bottom : float
        Response at zero (or very low) concentration.  Typically ~100 for
        percent-inhibition assays where vehicle = 100% viability.
    top : float
        Response at saturating concentration.  Typically ~0 for cytotoxicity.
    ic50 : float
        Concentration producing 50% of maximal effect.
    hill : float
        Hill slope (steepness of the sigmoid).  Positive values give a
        standard inhibition curve.

    Returns
    -------
    numpy.ndarray
        Predicted response at each concentration.
    """
    return bottom + (top - bottom) / (1.0 + (ic50 / concentration) ** hill)


# ---------------------------------------------------------------------------
# Concentration grid
# ---------------------------------------------------------------------------

def log_concentrations(low_uM=0.001, high_uM=100.0, n_points=10):
    """Return n_points log-spaced concentrations between low_uM and high_uM."""
    return np.logspace(np.log10(low_uM), np.log10(high_uM), n_points)


# ---------------------------------------------------------------------------
# Curve generators
# ---------------------------------------------------------------------------

def make_clean_sigmoid(rng):
    """
    Compound A: clean, well-behaved sigmoid.
    IC50 = 1 uM, Hill = 1.5, no noise.
    Represents an ideal result from a high-quality assay.
    """
    conc   = log_concentrations(0.001, 100.0, 10)
    true_params = dict(bottom=2.0, top=100.0, ic50=1.0, hill=1.5)
    resp   = four_pl(conc, **true_params)
    noise  = rng.normal(0, 1.5, size=len(conc))   # minimal noise
    resp   = np.clip(resp + noise, 0, 110)
    return conc, resp, true_params


def make_noisy_sigmoid(rng):
    """
    Compound B: same underlying curve as A but with substantial noise.
    Represents typical biological variability in a single-replicate screen.
    True IC50 = 1 uM -- fitting should still recover it reasonably well.
    """
    conc   = log_concentrations(0.001, 100.0, 10)
    true_params = dict(bottom=2.0, top=100.0, ic50=1.0, hill=1.5)
    resp   = four_pl(conc, **true_params)
    noise  = rng.normal(0, 12.0, size=len(conc))   # heavy noise
    resp   = np.clip(resp + noise, 0, 115)
    return conc, resp, true_params


def make_hook_effect(rng):
    """
    Compound C: hook effect -- activity peaks then drops at high concentration.
    Common cause: compound aggregation, solubility limit, or assay interference.
    A standard 4PL fit will perform poorly here; that is the lesson.
    """
    conc  = log_concentrations(0.001, 100.0, 10)
    # Rising inhibition phase (normal 4PL)
    resp  = four_pl(conc, bottom=5.0, top=95.0, ic50=0.5, hill=1.2)
    # Hook: at concentrations above 10 uM, response climbs back toward baseline
    hook_mask = conc > 10.0
    recovery  = 60.0 * (np.log10(conc[hook_mask]) - np.log10(10.0))
    resp[hook_mask] += recovery
    resp  = np.clip(resp, 0, 105)
    noise = rng.normal(0, 3.0, size=len(conc))
    resp  = np.clip(resp + noise, 0, 110)
    true_params = dict(bottom=5.0, top=95.0, ic50=0.5, hill=1.2,
                       note="hook_effect_above_10uM")
    return conc, resp, true_params


def make_incomplete_sigmoid(rng):
    """
    Compound D: incomplete sigmoid -- highest concentration not tested is still
    in the transition region.  The lower plateau is not reached, so IC50
    estimation will have high uncertainty.
    True IC50 = 50 uM (just outside the tested range).
    """
    conc   = log_concentrations(0.001, 10.0, 10)   # max dose only 10 uM
    true_params = dict(bottom=3.0, top=98.0, ic50=50.0, hill=1.0)
    resp   = four_pl(conc, **true_params)
    noise  = rng.normal(0, 3.0, size=len(conc))
    resp   = np.clip(resp + noise, 0, 110)
    return conc, resp, true_params


def make_inactive(rng):
    """
    Compound E: inactive compound -- flat line near 100% viability.
    No meaningful IC50 exists.  Fitting will produce a spurious value with
    enormous confidence intervals.
    """
    conc   = log_concentrations(0.001, 100.0, 10)
    resp   = np.full(len(conc), 98.0)
    noise  = rng.normal(0, 3.5, size=len(conc))
    resp   = np.clip(resp + noise, 80, 115)
    true_params = dict(bottom=None, top=None, ic50=None, hill=None,
                       note="inactive_no_ic50")
    return conc, resp, true_params


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_curve(compound_id, conc, resp, output_dir):
    """Save one compound's dose-response data as a CSV."""
    df = pd.DataFrame({
        "compound_id":      compound_id,
        "concentration_uM": np.round(conc, 6),
        "response_pct":     np.round(resp, 2),
    })
    path = Path(output_dir) / f"{compound_id}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}\n")

    curves = [
        ("compound_A_clean",      make_clean_sigmoid),
        ("compound_B_noisy",      make_noisy_sigmoid),
        ("compound_C_hook",       make_hook_effect),
        ("compound_D_incomplete", make_incomplete_sigmoid),
        ("compound_E_inactive",   make_inactive),
    ]

    all_dfs = []
    for cid, generator in curves:
        conc, resp, params = generator(rng)
        df = save_curve(cid, conc, resp, out)
        all_dfs.append(df)
        print(f"    True params: {params}\n")

    # Also save a combined file for convenience
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = out / "all_compounds.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined file saved: {combined_path.name}  ({len(combined)} rows total)")


if __name__ == "__main__":
    main()
