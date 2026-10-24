"""Download a real Gaia DR3 reference sample (needs internet + astroquery).

    python scripts/download_gaia.py --rows 20000

Default field: 1-degree cone around the Pleiades, a dense field (useful because close
pairs of stars make coordinate matching genuinely ambiguous).
"""
import argparse
from pathlib import Path

from astroquery.gaia import Gaia


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ra", type=float, default=56.75)
    p.add_argument("--dec", type=float, default=24.12)
    p.add_argument("--radius-deg", type=float, default=1.0)
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--out", default="data/raw/gaia_reference.csv")
    a = p.parse_args()

    query = f"""
        SELECT TOP {a.rows} source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {a.ra}, {a.dec}, {a.radius_deg}))
          AND parallax IS NOT NULL AND phot_g_mean_mag IS NOT NULL
    """
    df = Gaia.launch_job_async(query).get_results().to_pandas()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {len(df)} Gaia DR3 sources to {a.out}")


if __name__ == "__main__":
    main()
