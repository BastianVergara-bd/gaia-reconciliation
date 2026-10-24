"""Build the messy observation log used by the pipeline.

The log is SIMULATED: it takes real (or synthetic) Gaia reference rows and injects the
kinds of defects a real reconciliation job has to catch. Because defects are injected
on purpose, they are known, and the pipeline can be checked against them.

    python scripts/make_sample_data.py                          # fully synthetic demo
    python scripts/make_sample_data.py --reference data/raw/gaia_reference.csv   # real Gaia
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

COLS = ["source_id", "ra", "dec", "parallax", "parallax_error", "phot_g_mean_mag"]


def synthetic_reference(rng, n=3000, n_pairs=25) -> pd.DataFrame:
    ids = set()
    while len(ids) < n:
        ids.add(int(rng.integers(10**18, 6 * 10**18)))
    ids = np.array(sorted(ids), dtype="int64")
    dec = np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, n)))
    ref = pd.DataFrame({
        "source_id": ids, "ra": rng.uniform(0, 360, n), "dec": dec,
        "parallax": rng.gamma(2.0, 1.5, n) + 0.3,
        "parallax_error": rng.uniform(0.015, 0.05, n),
        "phot_g_mean_mag": rng.uniform(8, 17, n),
    })
    # close pairs (1.5-3 arcsec) so that coordinate matching can be genuinely ambiguous
    base = ref.sample(n_pairs, random_state=1).copy()
    base["dec"] += rng.uniform(1.5, 3.0, n_pairs) / 3600
    base["source_id"] = [int(rng.integers(10**18, 6 * 10**18)) for _ in range(n_pairs)]
    base["parallax"] = rng.gamma(2.0, 1.5, n_pairs) + 0.3
    return pd.concat([ref, base], ignore_index=True)


def close_pairs(ref, max_sep_arcsec=3.0):
    ra, dec = np.radians(ref["ra"].values), np.radians(ref["dec"].values)
    xyz = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    chord = 2 * np.sin(np.radians(max_sep_arcsec / 3600) / 2)
    return sorted(cKDTree(xyz).query_pairs(chord))


def make_log(ref, rng, n_obs=2500) -> pd.DataFrame:
    names = ["clean", "messy_id", "dr2", "truncated", "no_id_coords", "id_not_in_ref", "no_id_no_coords",
             "parallax_bad", "unit_arcsec", "unit_mislabel", "mag_bad", "coord_bad"]
    probs = [0.53, 0.14, 0.04, 0.03, 0.05, 0.02, 0.03, 0.04, 0.06, 0.02, 0.025, 0.015]
    picks = rng.choice(len(ref), size=min(n_obs, len(ref)), replace=False)
    scenarios = rng.choice(names, size=len(picks), p=np.array(probs) / sum(probs))
    templates = ["Gaia DR3 {}", " {} ", "GDR3-{}", "gaia dr3 {}", "0{}", "{}\u00a0"]
    rows = []
    for i, (k, s) in enumerate(zip(picks, scenarios)):
        r = ref.iloc[k]
        sid = int(ref["source_id"].values[k])   # not r.source_id: a mixed row Series is cast to float64 and loses digits
        ra = r.ra + rng.normal(0, 0.2) / 3600 / max(np.cos(np.radians(r.dec)), 0.1)
        dec = r.dec + rng.normal(0, 0.2) / 3600
        par = r.parallax + rng.normal(0, r.parallax_error)
        g = r.phot_g_mean_mag + rng.normal(0, 0.02)
        unit, target = "mas", str(sid)
        if s == "messy_id":
            target = templates[rng.integers(len(templates))].format(sid)
        elif s == "dr2":
            target = f"Gaia DR2 {int(rng.integers(10**18, 6 * 10**18))}"
            if rng.random() < 0.5:
                ra = dec = np.nan
        elif s == "truncated":
            target, ra, dec = f"{sid:.6e}", np.nan, np.nan
        elif s == "no_id_coords":
            target = None
        elif s == "id_not_in_ref":
            target = str(int(rng.integers(10**18, 6 * 10**18)))
        elif s == "no_id_no_coords":
            target, ra, dec = None, np.nan, np.nan
        elif s == "parallax_bad":
            par += rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
        elif s == "unit_arcsec":
            unit, par = "arcsec", par / 1000
        elif s == "unit_mislabel":
            par = par / 1000          # says mas, value is really arcsec
        elif s == "mag_bad":
            g += rng.uniform(0.3, 1.0)
        elif s == "coord_bad":
            dec += rng.uniform(8, 20) / 3600
        rows.append(dict(obs_id=f"OBS-{i:05d}", entry_type="observation", retracts_obs_id=None,
                         target_id=target, ra=ra, dec=dec, parallax=par, unit=unit, phot_g_mean_mag=g,
                         observed_at=str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))))[:10]))
    log = pd.DataFrame(rows)

    # ambiguous: no ID, coordinates halfway between two stars that are only a few arcsec apart
    for j, (a, b) in enumerate(close_pairs(ref)[:12]):
        ra_m, dec_m = (ref.ra[a] + ref.ra[b]) / 2, (ref.dec[a] + ref.dec[b]) / 2
        log.loc[len(log)] = dict(obs_id=f"AMB-{j:03d}", entry_type="observation", retracts_obs_id=None,
                                 target_id=None, ra=ra_m, dec=dec_m, parallax=2.0, unit="mas",
                                 phot_g_mean_mag=12.0, observed_at="2024-06-01")

    with_id = log[log["target_id"].notna() & (log["entry_type"] == "observation")]
    dups = with_id.sample(int(0.03 * len(log)), random_state=2).copy()      # exact repeats
    dups["obs_id"] = [f"DUP-{i:04d}" for i in range(len(dups))]
    victims = log.sample(int(0.02 * len(log)), random_state=3)["obs_id"].tolist()
    retr = pd.DataFrame({"obs_id": [f"RET-{i:04d}" for i in range(len(victims) + 3)],
                         "entry_type": "retraction",
                         "retracts_obs_id": victims + ["OBS-99998", "OBS-99999", "OBS-99997"]})  # last 3 = orphans
    out = pd.concat([log, dups, retr], ignore_index=True)
    return out.sample(frac=1, random_state=4).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default=None, help="Real Gaia CSV (from download_gaia.py)")
    p.add_argument("--out-dir", default="data/sample")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.reference:
        ref = pd.read_csv(a.reference, usecols=COLS, dtype={"source_id": "int64"}).dropna().reset_index(drop=True)
        ref_path = Path(a.reference)
    else:
        ref = synthetic_reference(rng)
        ref_path = out / "gaia_reference.csv"
        ref.to_csv(ref_path, index=False)
    log = make_log(ref, rng)
    log.to_csv(out / "observation_log.csv", index=False)
    print(f"Reference: {ref_path} ({len(ref)} stars)  |  Log: {out / 'observation_log.csv'} ({len(log)} rows)")


if __name__ == "__main__":
    main()
