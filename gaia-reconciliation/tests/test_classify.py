import pandas as pd
import pytest

from reconciliation.classify import detect_discrepancies, recommend_action
from reconciliation.matching import build_matches
from helpers import make_obs, make_ref, prepared


def classify(obs):
    res = recommend_action(detect_discrepancies(build_matches(prepared(obs), make_ref())))
    return res.iloc[0]["discrepancy"], res.iloc[0]["recommended_action"]


def test_consistent_row_is_accepted():
    assert classify(make_obs()) == ("NONE", "ACCEPT")


def test_small_parallax_difference_inside_tolerance_is_not_flagged():
    assert classify(make_obs(parallax=5.04))[0] == "NONE"          # 3 sigma = 0.06 mas


def test_large_parallax_difference_is_flagged_for_review():
    assert classify(make_obs(parallax=6.2)) == ("PARALLAX_MISMATCH", "REVIEW")


def test_value_in_arcsec_labelled_mas_is_a_suspected_unit_error():
    assert classify(make_obs(parallax=0.005, unit="mas")) == ("UNIT_ERROR_SUSPECTED", "REVIEW")


def test_value_correctly_labelled_arcsec_is_fine():
    assert classify(make_obs(parallax=0.005, unit="arcsec")) == ("NONE", "ACCEPT")


def test_magnitude_and_coordinate_mismatch():
    assert classify(make_obs(phot_g_mean_mag=12.5))[0] == "MAG_MISMATCH"
    assert classify(make_obs(dec=20.0 + 10 / 3600))[0] == "COORDINATE_MISMATCH"


def test_missing_parallax_is_incomplete_not_accepted():
    assert classify(make_obs(parallax=None)) == ("INCOMPLETE_MEASUREMENT", "REVIEW")


def test_duplicate_is_rejected():
    obs = pd.concat([make_obs(obs_id="A"), make_obs(obs_id="B")], ignore_index=True)
    res = recommend_action(detect_discrepancies(build_matches(prepared(obs), make_ref())))
    assert sorted(res["recommended_action"]) == ["ACCEPT", "REJECT_DUPLICATE"]


def test_coordinate_only_match_is_never_auto_accepted():
    assert classify(make_obs(target_id=None)) == ("NONE", "REVIEW")


def test_unmatched_rows_are_not_evaluated_and_marked_insufficient():
    assert classify(make_obs(target_id=None, ra=None, dec=None)) == ("NOT_EVALUATED", "INSUFFICIENT_DATA")
