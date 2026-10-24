"""Client-facing reports. IDs are written as text: Excel keeps only 15 significant
digits for numbers, which would silently corrupt 19-digit Gaia source_ids."""
from pathlib import Path

import pandas as pd

RESULT_COLUMNS = [
    "obs_id", "target_id", "matched_source_id", "match_status", "discrepancy",
    "recommended_action", "parallax_mas", "ref_parallax", "ref_parallax_error",
    "phot_g_mean_mag", "ref_phot_g_mean_mag", "coord_sep_arcsec", "n_candidates", "observed_at",
]


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reindex(columns=[c for c in RESULT_COLUMNS if c in df.columns]).copy()
    if "matched_source_id" in out:
        out["matched_source_id"] = out["matched_source_id"].astype("string")
    return out


def summarise(results: pd.DataFrame, retracted: pd.DataFrame, orphans: pd.DataFrame) -> pd.DataFrame:
    def count(col):
        c = results[col].value_counts().rename_axis("value").reset_index(name="rows")
        c.insert(0, "metric", col)
        return c
    extra = pd.DataFrame(
        {"metric": ["retracted_observations", "orphan_retractions", "total_active_rows"],
         "value": ["-", "-", "-"], "rows": [len(retracted), len(orphans), len(results)]}
    )
    return pd.concat([count("match_status"), count("discrepancy"), count("recommended_action"), extra], ignore_index=True)


def write_reports(results, retracted, orphans, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx, csv = out_dir / "reconciliation_report.xlsx", out_dir / "reconciliation_results.csv"

    printable = _printable(results)
    printable.to_csv(csv, index=False)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summarise(results, retracted, orphans).to_excel(writer, sheet_name="Summary", index=False)
        printable.to_excel(writer, sheet_name="All_Results", index=False)
        printable[results["recommended_action"] == "REVIEW"].to_excel(writer, sheet_name="Needs_Review", index=False)
        printable[results["recommended_action"] == "INSUFFICIENT_DATA"].to_excel(writer, sheet_name="Insufficient_Data", index=False)
        pd.concat([retracted.assign(kind="retracted_observation"), orphans.assign(kind="orphan_retraction")]).filter(
            ["kind", "obs_id", "retracts_obs_id", "target_id"]
        ).to_excel(writer, sheet_name="Retractions", index=False)
        for ws in writer.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(40, max(len(str(c.value or "")) for c in col) + 2)
    return {"xlsx": xlsx, "csv": csv}
