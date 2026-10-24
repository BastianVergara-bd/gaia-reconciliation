import pandas as pd
import pytest

from reconciliation.matching import angular_sep_arcsec, build_matches
from helpers import ID_A, ID_B, ID_C, make_obs, make_ref, prepared


def status_of(obs, ref=None):
    return build_matches(prepared(obs), ref if ref is not None else make_ref())["match_status"].iloc[0]


def test_exact_id_match():
    assert status_of(make_obs()) == "MATCHED_EXACT"


def test_cleaned_id_match_is_labelled_normalized():
    assert status_of(make_obs(target_id=f"Gaia DR3 {ID_A}")) == "MATCHED_NORMALIZED"


def test_valid_id_missing_from_reference_is_not_rescued_by_coordinates():
    # Coordinates sit exactly on a catalogue star, but the ID says otherwise: do not override the ID.
    assert status_of(make_obs(target_id="999999999999999999")) == "UNMATCHED_ID_NOT_IN_REFERENCE"


def test_single_coordinate_candidate_matches_when_id_is_missing():
    res = build_matches(prepared(make_obs(target_id=None)), make_ref())
    assert res.loc[0, "match_status"] == "MATCHED_BY_COORDINATES"
    assert int(res.loc[0, "matched_source_id"]) == ID_A


def test_two_candidates_inside_radius_is_ambiguous_and_never_forced():
    ref = make_ref([
        (ID_A, 10.0, 20.0, 5.0, 0.02, 12.0),
        (ID_B, 10.0, 20.0 + 1.5 / 3600, 4.0, 0.02, 15.0),   # 1.5 arcsec away
    ])
    res = build_matches(prepared(make_obs(target_id=None, dec=20.0 + 0.75 / 3600)), ref)
    assert res.loc[0, "match_status"] == "AMBIGUOUS_COORDINATES"
    assert res.loc[0, "n_candidates"] == 2
    assert pd.isna(res.loc[0, "matched_source_id"])


def test_dr2_id_without_coordinates_is_unsupported():
    assert status_of(make_obs(target_id=f"Gaia DR2 {ID_A}", ra=None, dec=None)) == "UNSUPPORTED_ID_NO_COORDS"


def test_no_id_no_coordinates_is_insufficient():
    assert status_of(make_obs(target_id=None, ra=None, dec=None)) == "INSUFFICIENT_DATA"


def test_row_count_is_preserved():
    obs = pd.concat([make_obs(obs_id="1"), make_obs(obs_id="2", target_id=None), make_obs(obs_id="3", target_id=None, ra=None, dec=None)], ignore_index=True)
    assert len(build_matches(prepared(obs), make_ref())) == 3


def test_duplicated_reference_ids_are_rejected():
    ref = make_ref([(ID_A, 10, 20, 5, 0.02, 12), (ID_A, 11, 21, 5, 0.02, 12)])
    with pytest.raises(ValueError):
        build_matches(prepared(make_obs()), ref)


def test_angular_separation_known_values():
    assert angular_sep_arcsec(10.0, 20.0, 10.0, 20.0 + 1 / 3600) == pytest.approx(1.0, rel=1e-6)
    assert angular_sep_arcsec(0.0, 0.0, 180.0, 0.0) == pytest.approx(180 * 3600)
