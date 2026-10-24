"""Matching observations against the Gaia reference catalogue.

Order of preference: (1) ID match, (2) coordinate match ONLY when there is no usable ID,
and only if exactly one reference star falls inside the radius. Ambiguous cases are
labelled, never forced.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config

VALUE_COLS = ["ra", "dec", "parallax", "parallax_error", "phot_g_mean_mag"]
MATCHED_STATUSES = {"MATCHED_EXACT", "MATCHED_NORMALIZED", "MATCHED_BY_COORDINATES"}


def unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation (haversine), in arcseconds."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    a = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * 3600


def match_by_id(obs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Left-merge on the normalised ID. validate= stops silent row duplication."""
    right = ref.rename(columns={"source_id": "norm_id", **{c: f"ref_{c}" for c in VALUE_COLS}})
    left = obs.copy()
    left["norm_id"] = left["norm_id"].astype("int64")
    merged = left.merge(right, on="norm_id", how="left", validate="many_to_one", indicator=True)
    both = (merged["_merge"] == "both").to_numpy()
    merged["norm_id"] = merged["norm_id"].astype("Int64")
    merged["matched_source_id"] = merged["norm_id"].where(both)
    merged["match_status"] = np.where(
        both,
        np.where(merged["id_was_cleaned"], "MATCHED_NORMALIZED", "MATCHED_EXACT"),
        "UNMATCHED_ID_NOT_IN_REFERENCE",
    )
    return merged.drop(columns="_merge")


def match_by_coordinates(obs: pd.DataFrame, ref: pd.DataFrame, radius_arcsec: float) -> pd.DataFrame:
    """Cone-search with a KD-tree on unit vectors. Exactly one candidate -> match."""
    out = obs.copy().reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    has_coords = (out["ra"].notna() & out["dec"].notna()).to_numpy()
    n_candidates = np.zeros(len(out), dtype="int64")
    ref_pos = np.full(len(out), -1, dtype="int64")

    if has_coords.any():
        tree = cKDTree(unit_vectors(ref["ra"].values, ref["dec"].values))
        chord = 2 * np.sin(np.radians(radius_arcsec / 3600) / 2)
        idx = np.flatnonzero(has_coords)
        hits = tree.query_ball_point(unit_vectors(out["ra"].values[idx], out["dec"].values[idx]), r=chord)
        for i, h in zip(idx, hits):
            n_candidates[i] = len(h)
            if len(h) == 1:
                ref_pos[i] = h[0]

    out["n_candidates"] = n_candidates
    unique = ref_pos >= 0
    for c in VALUE_COLS:
        out[f"ref_{c}"] = np.nan
        out.loc[unique, f"ref_{c}"] = ref[c].values[ref_pos[unique]]
    out["matched_source_id"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if unique.any():
        out.loc[unique, "matched_source_id"] = ref["source_id"].values[ref_pos[unique]]

    unsupported = out["id_reason"].isin(["dr2_release", "precision_lost"]).to_numpy()
    out["match_status"] = np.select(
        [n_candidates == 1, n_candidates > 1, has_coords, unsupported],
        ["MATCHED_BY_COORDINATES", "AMBIGUOUS_COORDINATES", "UNMATCHED_NO_COORD_CANDIDATE", "UNSUPPORTED_ID_NO_COORDS"],
        default="INSUFFICIENT_DATA",
    )
    return out


def build_matches(obs: pd.DataFrame, ref: pd.DataFrame, radius_arcsec: float = config.MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    if not ref["source_id"].is_unique:
        raise ValueError("Reference catalogue has duplicated source_id values")
    has_id = obs["norm_id"].notna()
    parts = []
    if has_id.any():
        parts.append(match_by_id(obs[has_id], ref))
    if (~has_id).any():
        parts.append(match_by_coordinates(obs[~has_id], ref, radius_arcsec))
    return pd.concat(parts, ignore_index=True)
