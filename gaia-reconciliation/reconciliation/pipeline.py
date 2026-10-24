"""End-to-end run:  python -m reconciliation.pipeline"""
import argparse
import logging

import pandas as pd

from . import config
from .classify import detect_discrepancies, recommend_action
from .cleaning import apply_retractions, clean_observations, flag_duplicates
from .matching import build_matches
from .report import write_reports

log = logging.getLogger("reconciliation")


def load_reference(path) -> pd.DataFrame:
    ref = pd.read_csv(path, usecols=config.REF_COLUMNS, dtype={"source_id": "int64"})
    return ref.dropna(subset=["ra", "dec"]).reset_index(drop=True)


def run(reference_path, observations_path, out_dir):
    ref = load_reference(reference_path)
    raw = pd.read_csv(observations_path, dtype={"target_id": "string"})
    log.info("Loaded %d reference stars and %d log rows", len(ref), len(raw))

    clean = clean_observations(raw)
    active, retracted, orphans = apply_retractions(clean)
    active = flag_duplicates(active)
    log.info("Retractions: %d observations cancelled, %d orphan retractions", len(retracted), len(orphans))

    matches = build_matches(active, ref)
    results = recommend_action(detect_discrepancies(matches))
    assert len(results) == len(active), "Row count changed during matching"

    paths = write_reports(results, retracted, orphans, out_dir)
    log.info("Action counts:\n%s", results["recommended_action"].value_counts().to_string())
    log.info("Reports written to %s", out_dir)
    return results, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/sample/gaia_reference.csv")
    parser.add_argument("--observations", default="data/sample/observation_log.csv")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.reference, args.observations, args.out)


if __name__ == "__main__":
    main()
