import pandas as pd
import pytest

from reconciliation.cleaning import apply_retractions, clean_observations, flag_duplicates, parse_source_id
from helpers import ID_A, make_obs


@pytest.mark.parametrize("raw", [
    str(ID_A), f" {ID_A} ", f"Gaia DR3 {ID_A}", f"gaia dr3 {ID_A}", f"GDR3-{ID_A}", f"0{ID_A}", f"{ID_A}\u00a0",
])
def test_messy_ids_normalise_to_same_value(raw):
    assert parse_source_id(raw) == (ID_A, "ok")


@pytest.mark.parametrize("raw, reason", [
    (None, "missing"), ("", "missing"), ("nan", "missing"),
    (f"Gaia DR2 {ID_A}", "dr2_release"),        # different catalogue release: never mapped
    ("4.472833e+18", "precision_lost"),          # digits already destroyed: never guessed
    ("12345", "malformed"),
])
def test_unsafe_ids_are_refused_not_guessed(raw, reason):
    value, why = parse_source_id(raw)
    assert value is None and why == reason


def test_arcsec_is_converted_to_mas_and_unknown_unit_is_not_assumed():
    df = pd.concat([make_obs(parallax=0.005, unit="arcsec"), make_obs(parallax=5.0, unit="furlongs")], ignore_index=True)
    out = clean_observations(df)
    assert out.loc[0, "parallax_mas"] == pytest.approx(5.0)
    assert pd.isna(out.loc[1, "parallax_mas"])


def test_id_was_cleaned_flag():
    out = clean_observations(pd.concat([make_obs(), make_obs(target_id=f"Gaia DR3 {ID_A}")], ignore_index=True))
    assert list(out["id_was_cleaned"]) == [False, True]


def test_retraction_cancels_its_target_and_orphans_are_reported():
    obs = pd.concat([
        make_obs(obs_id="OBS-1"), make_obs(obs_id="OBS-2"),
        pd.DataFrame([dict(obs_id="RET-1", entry_type="retraction", retracts_obs_id="OBS-1"),
                      dict(obs_id="RET-2", entry_type="retraction", retracts_obs_id="OBS-404")]),
    ], ignore_index=True)
    active, retracted, orphans = apply_retractions(obs)
    assert list(active["obs_id"]) == ["OBS-2"]
    assert list(retracted["obs_id"]) == ["OBS-1"]
    assert list(orphans["obs_id"]) == ["RET-2"]


def test_duplicates_keep_first_and_flag_the_rest():
    df = clean_observations(pd.concat([make_obs(obs_id="A"), make_obs(obs_id="B"), make_obs(obs_id="C", phot_g_mean_mag=13.0)], ignore_index=True))
    assert list(flag_duplicates(df)["is_duplicate"]) == [False, True, False]
