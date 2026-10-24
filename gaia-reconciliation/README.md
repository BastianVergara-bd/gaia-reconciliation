# Gaia Observation Reconciliation

A pandas pipeline that reconciles a **messy observation log** against the **Gaia DR3 catalogue**: it
cleans inconsistent IDs, matches records without forcing doubtful links, classifies every discrepancy
with a fixed taxonomy, and produces Excel/CSV reports a non-technical reader can act on.

The domain is astronomy, but the problem is the one found in financial reconciliation: match records
across sources with unreliable identifiers, compare values, and decide when the data is **not enough**
to support a claim.

| Reconciliation concept | Here |
|---|---|
| Invoices / purchase orders | Gaia reference catalogue (`source_id`, position, parallax, G magnitude) |
| Claims / payment records | Observation log with hand-typed IDs |
| Inconsistent IDs | `Gaia DR3 123…`, `GDR3-123…`, zero-padded, whitespace, DR2 IDs, IDs ruined by scientific notation |
| Multi-currency analysis | `mas` vs `arcsec` units (x1000), including mislabelled units |
| Underpayment check | Parallax difference beyond tolerance (3 sigma, with a floor) |
| Reversal matching | Retraction rows that cancel earlier observations (orphans reported) |
| Insufficient evidence | Rows with no usable ID *and* no usable coordinates |

## Data: what is real and what is simulated

* **Reference catalogue:** real Gaia DR3 when you run `scripts/download_gaia.py`. The sample committed in
  `data/sample/` is **synthetic data in the Gaia schema**, so the repo runs offline.
* **Observation log:** always **simulated**. `scripts/make_sample_data.py` injects known defects
  (messy IDs, DR2 IDs, truncated IDs, unit errors, wrong parallaxes, duplicates, retractions, close pairs),
  so the pipeline's behaviour can be checked against what was injected.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/make_sample_data.py                      # synthetic demo data
python -m reconciliation.pipeline                       # writes reports/
pytest -q
```

Using real Gaia data:

```bash
pip install -r requirements-data.txt
python scripts/download_gaia.py --rows 20000            # Pleiades field, needs internet
python scripts/make_sample_data.py --reference data/raw/gaia_reference.csv
python -m reconciliation.pipeline --reference data/raw/gaia_reference.csv
```

## How it works

1. **Clean** (`cleaning.py`): parse IDs, convert units to mas, coerce types. IDs that cannot be recovered
   safely are *refused, not guessed*: DR2 IDs belong to another release and scientific-notation IDs have
   lost digits.
2. **Retractions and duplicates:** cancel retracted observations, report orphan retractions, flag repeats.
3. **Match** (`matching.py`): merge on normalised ID with `validate="many_to_one"`. Only rows with **no
   usable ID** fall back to a KD-tree cone search, and only a single candidate counts as a match; two or
   more is `AMBIGUOUS_COORDINATES`. A valid ID missing from the catalogue is never rescued by position.
4. **Classify** (`classify.py`): first-hit-wins taxonomy: `DUPLICATE_ENTRY`, `COORDINATE_MISMATCH`,
   `INCOMPLETE_MEASUREMENT`, `UNIT_ERROR_SUSPECTED`, `PARALLAX_MISMATCH`, `MAG_MISMATCH`, `NONE`.
   Each row gets an action: `ACCEPT`, `REVIEW`, `REJECT_DUPLICATE` or `INSUFFICIENT_DATA`.
5. **Report** (`report.py`): Excel (Summary, All_Results, Needs_Review, Insufficient_Data, Retractions) and CSV.

## Results on the bundled sample (2,537 active rows after 50 retractions)

| Action | Rows |
|---|---|
| ACCEPT | 1,725 |
| REVIEW | 529 |
| INSUFFICIENT_DATA | 214 |
| REJECT_DUPLICATE | 69 |

Match statuses: 1,734 exact, 312 normalised, 204 by coordinates, 15 ambiguous, 58 ID not in reference,
214 unsupported or insufficient. Discrepancies found: 113 parallax, 65 magnitude, 43 suspected unit errors,
35 coordinate mismatches, 69 duplicates.

## Design decisions

* **Refuse over guess.** A wrong match is worse than an unmatched row, so every uncertain case gets its own label.
* **IDs are 19-digit integers.** They are kept as `int64`/`Int64` and written as text in Excel, which only holds
  15 significant digits. A float cast anywhere in the chain silently corrupts them (I hit this while generating
  the data, which is why `tests/` checks round trips).
* **Thresholds live in `config.py`**, so each rule can be tuned and tested in one place.
* **Coordinate-only matches are never auto-accepted.** They go to `REVIEW`.

## Limitations and next steps

* Position matching ignores Gaia proper motion and epoch; a real survey would propagate positions first.
* Tested on thousands of rows. For millions, I would read Parquet in chunks and replace the Python loop in
  the cone search with a vectorised query.
* PDF export of the report is not implemented.

## Project layout

```
reconciliation/   config, cleaning, matching, classify, report, pipeline
scripts/          download_gaia.py, make_sample_data.py
tests/            pytest suite (cleaning, matching, classification)
data/sample/      small synthetic dataset so the repo runs offline
```
