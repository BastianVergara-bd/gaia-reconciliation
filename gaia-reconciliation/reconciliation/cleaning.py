"""Cleaning and normalisation of the messy observation log."""
import re

import numpy as np
import pandas as pd

from . import config

_SCIENTIFIC = re.compile(r"\d\.?\d*e[+-]?\d+")


def parse_source_id(raw):
    """Try to recover a Gaia DR3 source_id from a messy value.

    Returns (source_id | None, reason). We refuse to guess: a DR2 id belongs to a
    different catalogue release, and an id written in scientific notation has lost
    digits, so neither can support a match.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None, "missing"
    text = str(raw).strip().lower()
    if text in {"", "nan", "none", "null", "<na>"}:
        return None, "missing"
    if "dr2" in text:
        return None, "dr2_release"
    if _SCIENTIFIC.search(text):
        return None, "precision_lost"
    text = re.sub(r"e?dr3|gaia", "", text)
    digits = re.sub(r"\D", "", text).lstrip("0")
    if not (config.GAIA_ID_MIN_DIGITS <= len(digits) <= config.GAIA_ID_MAX_DIGITS):
        return None, "malformed"
    return int(digits), "ok"


def clean_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise IDs, numeric columns, units and dates. Never drops rows."""
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    parsed = [parse_source_id(v) for v in out["target_id"]]
    out["norm_id"] = pd.array([p[0] for p in parsed], dtype="Int64")
    out["id_reason"] = [p[1] for p in parsed]
    raw_text = out["target_id"].astype("string")   # unstripped: padding counts as cleaning
    out["id_was_cleaned"] = (out["norm_id"].notna() & (raw_text != out["norm_id"].astype("string"))).astype(bool)

    for col in ["ra", "dec", "parallax", "phot_g_mean_mag"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    unit = out["unit"].astype("string").str.strip().str.lower()
    factor = unit.map(config.VALID_UNITS).astype("float64")   # unknown unit -> NaN, never assumed
    out["parallax_mas"] = out["parallax"] * factor
    out["observed_at"] = pd.to_datetime(out["observed_at"], errors="coerce")
    return out


def apply_retractions(df: pd.DataFrame):
    """Pair retraction rows with the observations they cancel (reversal matching).

    Returns (active_observations, retracted_observations, orphan_retractions).
    An orphan is a retraction pointing at an observation that does not exist.
    """
    is_obs = df["entry_type"] == "observation"
    retractions = df[df["entry_type"] == "retraction"]
    targets = set(retractions["retracts_obs_id"].dropna())
    retracted_mask = is_obs & df["obs_id"].isin(targets)
    orphans = retractions[~retractions["retracts_obs_id"].isin(set(df.loc[is_obs, "obs_id"]))]
    return (
        df[is_obs & ~retracted_mask].reset_index(drop=True),
        df[retracted_mask].reset_index(drop=True),
        orphans.reset_index(drop=True),
    )


def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag repeated entries (same star, same time, same magnitude). First one is kept."""
    out = df.copy()
    keyed = out["norm_id"].notna()
    dup = out.duplicated(subset=["norm_id", "observed_at", "phot_g_mean_mag"], keep="first")
    out["is_duplicate"] = (dup & keyed).astype(bool)
    return out
