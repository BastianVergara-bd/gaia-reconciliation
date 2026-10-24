"""Tunable thresholds. Every decision rule in the pipeline reads from here."""

MATCH_RADIUS_ARCSEC = 2.0      # cone radius for coordinate fallback matching
COORD_MISMATCH_ARCSEC = 2.0    # ID-matched row whose position is farther than this is suspicious
PARALLAX_SIGMA = 3.0           # tolerance = PARALLAX_SIGMA * reference parallax_error ...
PARALLAX_FLOOR_MAS = 0.05      # ... but never tighter than this floor
MAG_TOLERANCE = 0.10           # max |G_obs - G_ref| in magnitudes
VALID_UNITS = {"mas": 1.0, "arcsec": 1000.0}   # factor to convert to milliarcseconds
GAIA_ID_MIN_DIGITS = 17
GAIA_ID_MAX_DIGITS = 19

REF_COLUMNS = ["source_id", "ra", "dec", "parallax", "parallax_error", "phot_g_mean_mag"]
