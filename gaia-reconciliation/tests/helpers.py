"""Small builders so each test shows exactly the rows it cares about."""
import pandas as pd

from reconciliation.cleaning import clean_observations, flag_duplicates

ID_A, ID_B, ID_C = 4472832130942575872, 4472832130942575873, 1050646686707294464


def make_ref(rows=None) -> pd.DataFrame:
    rows = rows or [
        (ID_A, 10.0, 20.0, 5.000, 0.020, 12.0),
        (ID_C, 200.0, -30.0, 2.000, 0.020, 14.0),
    ]
    return pd.DataFrame(rows, columns=["source_id", "ra", "dec", "parallax", "parallax_error", "phot_g_mean_mag"])


def make_obs(**overrides) -> pd.DataFrame:
    row = dict(obs_id="OBS-1", entry_type="observation", retracts_obs_id=None, target_id=str(ID_A),
               ra=10.0, dec=20.0, parallax=5.0, unit="mas", phot_g_mean_mag=12.0, observed_at="2024-01-01")
    row.update(overrides)
    return pd.DataFrame([row])


def prepared(df: pd.DataFrame) -> pd.DataFrame:
    return flag_duplicates(clean_observations(df))
