"""Discrepancy taxonomy and recommended actions.

Discrepancy types (checked in this order, first hit wins):
  DUPLICATE_ENTRY       same star/time/magnitude logged more than once
  COORDINATE_MISMATCH   ID matched, but position is far from the catalogue position
  INCOMPLETE_MEASUREMENT matched, but parallax or unit is missing/unknown
  UNIT_ERROR_SUSPECTED  value is off by exactly the mas<->arcsec factor (x1000)
  PARALLAX_MISMATCH     |obs - ref| above tolerance
  MAG_MISMATCH          |G_obs - G_ref| above tolerance
  NONE                  matched and consistent
  NOT_EVALUATED         row never matched, so nothing to compare
"""
import numpy as np
import pandas as pd

from . import config
from .matching import MATCHED_STATUSES, angular_sep_arcsec


def detect_discrepancies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    matched = out["match_status"].isin(MATCHED_STATUSES).to_numpy()

    sep = angular_sep_arcsec(out["ra"], out["dec"], out["ref_ra"], out["ref_dec"])
    out["coord_sep_arcsec"] = sep

    tol = np.maximum(config.PARALLAX_SIGMA * out["ref_parallax_error"], config.PARALLAX_FLOOR_MAS)
    par = out["parallax_mas"]
    ref_par = out["ref_parallax"]
    par_off = (par - ref_par).abs() > tol
    unit_err = par_off & (
        ((par * 1000 - ref_par).abs() <= tol) | ((par / 1000 - ref_par).abs() <= tol)
    )
    mag_off = (out["phot_g_mean_mag"] - out["ref_phot_g_mean_mag"]).abs() > config.MAG_TOLERANCE

    conditions = [
        out["is_duplicate"].to_numpy(),
        matched & (sep > config.COORD_MISMATCH_ARCSEC).fillna(False).to_numpy(),
        matched & par.isna().to_numpy(),
        matched & unit_err.fillna(False).to_numpy(),
        matched & par_off.fillna(False).to_numpy(),
        matched & mag_off.fillna(False).to_numpy(),
        matched,
    ]
    labels = [
        "DUPLICATE_ENTRY", "COORDINATE_MISMATCH", "INCOMPLETE_MEASUREMENT",
        "UNIT_ERROR_SUSPECTED", "PARALLAX_MISMATCH", "MAG_MISMATCH", "NONE",
    ]
    out["discrepancy"] = np.select(conditions, labels, default="NOT_EVALUATED")
    return out


def recommend_action(df: pd.DataFrame) -> pd.DataFrame:
    """ACCEPT / REVIEW / REJECT_DUPLICATE / INSUFFICIENT_DATA.

    A coordinate-only match is never auto-accepted: it is a lower-confidence link.
    """
    out = df.copy()
    status, disc = out["match_status"], out["discrepancy"]
    insufficient = status.isin(["INSUFFICIENT_DATA", "UNSUPPORTED_ID_NO_COORDS", "UNMATCHED_NO_COORD_CANDIDATE"])
    out["recommended_action"] = np.select(
        [
            disc == "DUPLICATE_ENTRY",
            insufficient,
            status.isin(["MATCHED_EXACT", "MATCHED_NORMALIZED"]) & (disc == "NONE"),
        ],
        ["REJECT_DUPLICATE", "INSUFFICIENT_DATA", "ACCEPT"],
        default="REVIEW",
    )
    return out
