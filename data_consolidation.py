"""
data_consolidation.py — Multi-source exoplanet data consolidation pipeline.

Merges data from:
  1. NASA Exoplanet Archive (PS table) — base dataset
  2. HEC (Habitable Exoplanets Catalog)    — habitability labels
  3. Gaia DR3                              — fills missing stellar parameters
  4. SIMBAD                                — fills spectral type & remaining gaps

Output: data/consolidated_exoplanets.csv
"""

import argparse
import io
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"

NASA_CACHE = DATA_DIR / "nasa_ps.csv"
GAIA_CACHE = DATA_DIR / "gaia_stellar.csv"
SIMBAD_CACHE = DATA_DIR / "simbad_stellar.csv"
HEC_FILE = DATA_DIR / "hwc.csv"
OUTPUT_FILE = DATA_DIR / "consolidated_exoplanets.csv"

# Physical constants (SI)
G_SI = 6.674e-11          # m^3 kg^-1 s^-2
M_EARTH = 5.972e24        # kg
R_EARTH = 6.371e6         # m
R_SUN = 6.957e8           # m

# HEC columns that provide unique value (not duplicated in NASA PS)
HEC_UNIQUE_COLS = [
    "P_HABITABLE", "P_ESI", "P_TYPE", "P_TYPE_TEMP",
    "P_HABZONE_OPT", "P_HABZONE_CON",
    "P_ESCAPE", "P_POTENTIAL", "P_GRAVITY", "P_DENSITY", "P_HILL_SPHERE",
    "P_FLUX", "P_FLUX_MIN", "P_FLUX_MAX",
    "P_TEMP_EQUIL", "P_TEMP_EQUIL_MIN", "P_TEMP_EQUIL_MAX",
    "P_TEMP_SURF", "P_TEMP_SURF_MIN", "P_TEMP_SURF_MAX",
    "P_DISTANCE", "P_DISTANCE_EFF",
    "S_HZ_OPT_MIN", "S_HZ_OPT_MAX",
    "S_HZ_CON_MIN", "S_HZ_CON_MAX",
    "S_HZ_CON0_MIN", "S_HZ_CON0_MAX",
    "S_HZ_CON1_MIN", "S_HZ_CON1_MAX",
    "S_SNOW_LINE", "S_TIDAL_LOCK", "S_ABIO_ZONE",
]

# Gaia → NASA fill mapping
GAIA_FILL_MAP = {
    "st_teff":  "gaia_teff_gspphot",
    "st_lum":   "gaia_lum_flame",
    "st_mass":  "gaia_mass_flame",
    "st_rad":   "gaia_radius_flame",
    "st_logg":  "gaia_logg_gspphot",
    "st_met":   "gaia_mh_gspphot",
    "st_age":   "gaia_age_flame",
}

# Metadata columns to drop from final output
DROP_COLS_PATTERNS = [
    "rowupdate", "pl_pubdate", "releasedate", "*_lim", "*_flag",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_planet_name(name: str) -> str:
    """Normalize a planet name for cross-catalog matching."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    # Normalize unicode dashes and special chars
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    # Remove trailing period
    s = s.rstrip(".")
    return s


def tap_query(url: str, adql: str, cache_path: Path | None = None,
              force: bool = False, max_retries: int = 3) -> pd.DataFrame:
    """Execute a TAP sync query and return a DataFrame. Caches to CSV."""
    if cache_path and cache_path.exists() and not force:
        log.info(f"Loading cached {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }

    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"TAP query attempt {attempt}/{max_retries} → {url}")
            log.debug(f"ADQL: {adql[:200]}...")
            resp = requests.post(url, data=params, timeout=300)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            log.info(f"  → {len(df)} rows, {len(df.columns)} columns")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                log.info(f"  Cached to {cache_path}")
            return df
        except Exception as e:
            log.warning(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                log.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Step 1: NASA Exoplanet Archive
# ---------------------------------------------------------------------------
def download_nasa_ps(force: bool = False) -> pd.DataFrame:
    """Download the NASA PS table (default_flag=1) via TAP."""
    adql = "SELECT * FROM ps WHERE default_flag = 1"
    df = tap_query(NASA_TAP_URL, adql, NASA_CACHE, force=force)
    log.info(f"NASA PS: {len(df)} planets, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2: HEC (Habitable Exoplanets Catalog)
# ---------------------------------------------------------------------------
def load_hec() -> pd.DataFrame:
    """Load the HEC dataset from local CSV."""
    df = pd.read_csv(HEC_FILE, low_memory=False)
    log.info(f"HEC: {len(df)} rows, {len(df.columns)} columns")
    return df


def merge_hec(nasa: pd.DataFrame, hec: pd.DataFrame) -> pd.DataFrame:
    """Left-join HEC habitability columns onto NASA base."""
    # Build normalized name keys
    nasa = nasa.copy()
    hec = hec.copy()
    nasa["_join_name"] = nasa["pl_name"].apply(normalize_planet_name)
    hec["_join_name"] = hec["P_NAME"].apply(normalize_planet_name)

    # Select only the unique HEC columns + join key
    available_cols = [c for c in HEC_UNIQUE_COLS if c in hec.columns]
    hec_subset = hec[["_join_name"] + available_cols].copy()

    # Deduplicate HEC on join key (keep first occurrence)
    hec_subset = hec_subset.drop_duplicates(subset="_join_name", keep="first")

    merged = nasa.merge(hec_subset, on="_join_name", how="left")
    matched = merged[available_cols[0]].notna().sum() if available_cols else 0
    log.info(f"HEC merge: {matched}/{len(nasa)} planets matched ({matched/len(nasa)*100:.1f}%)")

    merged.drop(columns=["_join_name"], inplace=True)
    return merged


# ---------------------------------------------------------------------------
# Step 3: Gaia DR3
# ---------------------------------------------------------------------------
def query_gaia(nasa: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Query Gaia DR3 for stellar parameters using source IDs from NASA."""
    if GAIA_CACHE.exists() and not force:
        log.info(f"Loading cached {GAIA_CACHE}")
        return pd.read_csv(GAIA_CACHE, low_memory=False)

    # Extract numeric Gaia DR3 source IDs
    gaia_col = None
    for col_name in ["gaia_id", "gaia_dr3_id"]:
        if col_name in nasa.columns:
            gaia_col = col_name
            break

    if gaia_col is None:
        log.warning("No Gaia ID column found in NASA data; skipping Gaia query.")
        return pd.DataFrame()

    raw_ids = nasa[gaia_col].dropna().unique()
    # Extract numeric source_id from strings like "Gaia DR3 1234567890"
    source_ids = []
    for raw in raw_ids:
        s = str(raw).strip()
        match = re.search(r"(\d{5,})", s)
        if match:
            source_ids.append(int(match.group(1)))

    log.info(f"Gaia: {len(source_ids)} unique source IDs to query")
    if not source_ids:
        return pd.DataFrame()

    # Columns from gaiadr3.gaia_source
    main_cols = (
        "source_id, teff_gspphot, logg_gspphot, mh_gspphot, "
        "radial_velocity, radial_velocity_error, parallax, parallax_error, "
        "phot_g_mean_mag"
    )
    # Columns from gaiadr3.astrophysical_parameters (FLAME pipeline)
    astro_cols = "source_id, lum_flame, mass_flame, radius_flame, age_flame"

    batch_size = 500
    main_results = []
    astro_results = []

    for i in range(0, len(source_ids), batch_size):
        batch = source_ids[i : i + batch_size]
        id_list = ", ".join(str(sid) for sid in batch)
        batch_num = i // batch_size + 1
        total_batches = (len(source_ids) + batch_size - 1) // batch_size

        # Query main gaia_source table
        adql_main = f"SELECT {main_cols} FROM gaiadr3.gaia_source WHERE source_id IN ({id_list})"
        log.info(f"Gaia batch {batch_num}/{total_batches}: {len(batch)} IDs (gaia_source)")
        try:
            df = tap_query(GAIA_TAP_URL, adql_main, cache_path=None, force=True)
            main_results.append(df)
        except Exception as e:
            log.error(f"Gaia gaia_source batch failed: {e}")
        time.sleep(1)

        # Query astrophysical_parameters table
        adql_astro = f"SELECT {astro_cols} FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({id_list})"
        log.info(f"Gaia batch {batch_num}/{total_batches}: {len(batch)} IDs (astrophysical_parameters)")
        try:
            df = tap_query(GAIA_TAP_URL, adql_astro, cache_path=None, force=True)
            astro_results.append(df)
        except Exception as e:
            log.error(f"Gaia astrophysical_parameters batch failed: {e}")
        time.sleep(1)

    if not main_results:
        log.warning("No Gaia results obtained.")
        return pd.DataFrame()

    # Combine main table results
    gaia_df = pd.concat(main_results, ignore_index=True)
    gaia_df = gaia_df.drop_duplicates(subset="source_id", keep="first")

    # Combine and merge astrophysical params
    if astro_results:
        astro_df = pd.concat(astro_results, ignore_index=True)
        astro_df = astro_df.drop_duplicates(subset="source_id", keep="first")
        gaia_df = gaia_df.merge(astro_df, on="source_id", how="left")

    # Cache
    GAIA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gaia_df.to_csv(GAIA_CACHE, index=False)
    log.info(f"Gaia: {len(gaia_df)} stellar records retrieved and cached.")
    return gaia_df


def merge_gaia(df: pd.DataFrame, gaia_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Gaia stellar params and fill missing NASA values."""
    if gaia_df.empty:
        log.info("No Gaia data to merge.")
        return df

    df = df.copy()

    # Build source_id column in NASA data for joining
    gaia_col = None
    for col_name in ["gaia_id", "gaia_dr3_id"]:
        if col_name in df.columns:
            gaia_col = col_name
            break

    if gaia_col is None:
        return df

    def extract_source_id(raw):
        if pd.isna(raw):
            return np.nan
        match = re.search(r"(\d{5,})", str(raw).strip())
        return int(match.group(1)) if match else np.nan

    df["_gaia_source_id"] = df[gaia_col].apply(extract_source_id)

    # Prefix Gaia columns to avoid collision, then merge
    gaia_renamed = gaia_df.rename(
        columns={c: f"gaia_{c}" if c != "source_id" else c for c in gaia_df.columns}
    )
    merged = df.merge(
        gaia_renamed, left_on="_gaia_source_id", right_on="source_id", how="left"
    )
    merged.drop(columns=["source_id", "_gaia_source_id"], inplace=True, errors="ignore")

    # Fill missing NASA stellar values from Gaia
    fill_count = 0
    for nasa_col, gaia_col_name in GAIA_FILL_MAP.items():
        if nasa_col in merged.columns and gaia_col_name in merged.columns:
            mask = merged[nasa_col].isna() & merged[gaia_col_name].notna()
            merged.loc[mask, nasa_col] = merged.loc[mask, gaia_col_name]
            filled = mask.sum()
            if filled > 0:
                log.info(f"  Gaia filled {filled} missing values for {nasa_col}")
                fill_count += filled

    log.info(f"Gaia merge: {fill_count} total values filled across all columns")
    merged["source_gaia"] = merged.get("gaia_teff_gspphot", pd.Series(dtype=float)).notna()
    return merged


# ---------------------------------------------------------------------------
# Step 4: SIMBAD
# ---------------------------------------------------------------------------
def query_simbad(hostnames: list[str], force: bool = False) -> pd.DataFrame:
    """Query SIMBAD for spectral types using host star names."""
    if SIMBAD_CACHE.exists() and not force:
        log.info(f"Loading cached {SIMBAD_CACHE}")
        return pd.read_csv(SIMBAD_CACHE, low_memory=False)

    unique_hosts = sorted(set(h for h in hostnames if pd.notna(h) and str(h).strip()))
    log.info(f"SIMBAD: querying {len(unique_hosts)} unique host stars")

    if not unique_hosts:
        return pd.DataFrame()

    batch_size = 200
    all_results = []

    for i in range(0, len(unique_hosts), batch_size):
        batch = unique_hosts[i : i + batch_size]
        # Escape single quotes in star names
        escaped = [name.replace("'", "''") for name in batch]
        in_clause = ", ".join(f"'{name}'" for name in escaped)

        adql = (
            "SELECT b.main_id, b.sp_type, b.rvz_radvel, b.rvz_type, i.id AS query_id "
            "FROM ident AS i "
            "JOIN basic AS b ON i.oidref = b.oid "
            f"WHERE i.id IN ({in_clause})"
        )

        log.info(f"SIMBAD batch {i // batch_size + 1}/{(len(unique_hosts) + batch_size - 1) // batch_size}: {len(batch)} stars")
        try:
            df = tap_query(SIMBAD_TAP_URL, adql, cache_path=None, force=True)
            all_results.append(df)
        except Exception as e:
            log.error(f"SIMBAD batch failed: {e}")
        time.sleep(1)  # rate limiting

    if not all_results:
        log.warning("No SIMBAD results obtained.")
        return pd.DataFrame()

    simbad_df = pd.concat(all_results, ignore_index=True)

    # Clean up query_id for joining back
    if "query_id" in simbad_df.columns:
        simbad_df["query_id"] = simbad_df["query_id"].str.strip()
        simbad_df = simbad_df.drop_duplicates(subset="query_id", keep="first")

    SIMBAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    simbad_df.to_csv(SIMBAD_CACHE, index=False)
    log.info(f"SIMBAD: {len(simbad_df)} stellar records retrieved and cached.")
    return simbad_df


def merge_simbad(df: pd.DataFrame, simbad_df: pd.DataFrame) -> pd.DataFrame:
    """Merge SIMBAD spectral type data into the consolidated DataFrame."""
    if simbad_df.empty or "query_id" not in simbad_df.columns:
        log.info("No SIMBAD data to merge.")
        return df

    df = df.copy()
    simbad_subset = simbad_df[["query_id", "sp_type", "rvz_radvel"]].copy()
    simbad_subset = simbad_subset.rename(columns={
        "sp_type": "simbad_sp_type",
        "rvz_radvel": "simbad_radvel",
    })

    merged = df.merge(simbad_subset, left_on="hostname", right_on="query_id", how="left")
    merged.drop(columns=["query_id"], inplace=True, errors="ignore")

    # Fill spectral type if missing
    if "st_spectype" in merged.columns and "simbad_sp_type" in merged.columns:
        mask = merged["st_spectype"].isna() & merged["simbad_sp_type"].notna()
        merged.loc[mask, "st_spectype"] = merged.loc[mask, "simbad_sp_type"]
        log.info(f"SIMBAD filled {mask.sum()} missing spectral types")

    merged["source_simbad"] = merged["simbad_sp_type"].notna()
    return merged


# ---------------------------------------------------------------------------
# Step 5: Derived features
# ---------------------------------------------------------------------------
def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical derived features for planets missing them."""
    df = df.copy()

    # --- Surface gravity (Earth units) ---
    # g = G * M / R^2 ; in Earth units: g_earth = (M/M_earth) / (R/R_earth)^2
    if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
        mask = df.get("P_GRAVITY", pd.Series(dtype=float)).isna()
        mass_e = df["pl_bmasse"]  # Earth masses
        rad_e = df["pl_rade"]     # Earth radii
        gravity = mass_e / (rad_e ** 2)
        if "P_GRAVITY" not in df.columns:
            df["P_GRAVITY"] = np.nan
        df.loc[mask & mass_e.notna() & rad_e.notna(), "P_GRAVITY"] = gravity[mask & mass_e.notna() & rad_e.notna()]
        filled = (mask & mass_e.notna() & rad_e.notna()).sum()
        if filled > 0:
            log.info(f"Computed gravity for {filled} planets")

    # --- Density (Earth units: rho/rho_earth) ---
    # rho = M / (4/3 * pi * R^3) ; in Earth units: rho_earth = (M/M_earth) / (R/R_earth)^3
    if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
        mask = df.get("P_DENSITY", pd.Series(dtype=float)).isna()
        density = mass_e / (rad_e ** 3)
        if "P_DENSITY" not in df.columns:
            df["P_DENSITY"] = np.nan
        valid = mask & mass_e.notna() & rad_e.notna()
        df.loc[valid, "P_DENSITY"] = density[valid]
        filled = valid.sum()
        if filled > 0:
            log.info(f"Computed density for {filled} planets")

    # --- Escape velocity (Earth units) ---
    # v_esc = sqrt(2GM/R) ; in Earth units: v_esc_earth = sqrt((M/M_earth) / (R/R_earth))
    if "pl_bmasse" in df.columns and "pl_rade" in df.columns:
        mask = df.get("P_ESCAPE", pd.Series(dtype=float)).isna()
        esc_vel = np.sqrt(mass_e / rad_e)
        if "P_ESCAPE" not in df.columns:
            df["P_ESCAPE"] = np.nan
        valid = mask & mass_e.notna() & rad_e.notna() & (rad_e > 0)
        df.loc[valid, "P_ESCAPE"] = esc_vel[valid]
        filled = valid.sum()
        if filled > 0:
            log.info(f"Computed escape velocity for {filled} planets")

    # --- Equilibrium temperature (K) ---
    # T_eq = T_star * sqrt(R_star / (2 * a))
    # where a = semi-major axis in AU, R_star in solar radii, T_star in K
    if all(c in df.columns for c in ["st_teff", "st_rad", "pl_orbsmax"]):
        if "pl_eqt" not in df.columns:
            df["pl_eqt"] = np.nan
        mask = df["pl_eqt"].isna()
        t_star = df["st_teff"]
        r_star_au = df["st_rad"] * R_SUN / 1.496e11  # solar radii → AU
        a = df["pl_orbsmax"]  # AU
        valid = mask & t_star.notna() & r_star_au.notna() & a.notna() & (a > 0)
        t_eq = t_star * np.sqrt(r_star_au / (2 * a))
        df.loc[valid, "pl_eqt"] = t_eq[valid]
        filled = valid.sum()
        if filled > 0:
            log.info(f"Computed equilibrium temperature for {filled} planets")

    # --- Habitable Zone boundaries (Kopparapu et al. 2013) ---
    # Conservative HZ inner/outer edges based on stellar Teff and luminosity
    if "st_teff" in df.columns and "st_lum" in df.columns:
        _compute_hz(df)

    # --- Earth Similarity Index (ESI) ---
    _compute_esi(df)

    # --- Planet type classification ---
    _classify_planet_type(df)

    return df


def _compute_hz(df: pd.DataFrame) -> None:
    """Compute habitable zone boundaries using Kopparapu et al. (2013)."""
    # Coefficients for Recent Venus (inner) and Early Mars (outer) — optimistic HZ
    # S_eff = S_eff_sun + a*T + b*T^2 + c*T^3 + d*T^4, where T = Teff - 5780
    hz_coeffs = {
        "inner_opt": (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
        "outer_opt": (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
        "inner_con": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
        "outer_con": (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    }

    teff = df["st_teff"]
    # Luminosity: NASA stores as log10(L/L_sun)
    lum_col = "st_lum"
    if lum_col in df.columns:
        lum = 10 ** df[lum_col].astype(float)  # convert log to linear
    else:
        return

    t_diff = teff - 5780.0

    import warnings
    for key, (s0, a, b, c, d) in hz_coeffs.items():
        col_name = f"hz_{key}"
        if col_name in df.columns:
            mask = df[col_name].isna()
        else:
            df[col_name] = np.nan
            mask = pd.Series(True, index=df.index)

        s_eff = s0 + a * t_diff + b * t_diff**2 + c * t_diff**3 + d * t_diff**4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hz_dist = np.sqrt(lum / s_eff)  # AU

        valid = mask & teff.notna() & lum.notna()
        df.loc[valid, col_name] = hz_dist[valid]

    # Habitable zone flag: is the planet's semi-major axis within the optimistic HZ?
    if "pl_orbsmax" in df.columns:
        a = df["pl_orbsmax"]
        in_hz = (a >= df["hz_inner_opt"]) & (a <= df["hz_outer_opt"])
        df["in_habitable_zone"] = in_hz
        log.info(f"Planets in habitable zone: {in_hz.sum()}")


def _compute_esi(df: pd.DataFrame) -> None:
    """Compute Earth Similarity Index (ESI)."""
    # ESI = product of (1 - |x - x_earth| / (x + x_earth))^(w/n) for each parameter
    # Parameters: radius, density, escape velocity, surface temp
    # Earth values: R=1, rho=1, v_esc=1 (all in Earth units), T_surf=288 K

    if "P_ESI" not in df.columns:
        df["P_ESI"] = np.nan

    mask = df["P_ESI"].isna()

    # We need at least radius and one more parameter
    params = []
    if "pl_rade" in df.columns:
        params.append(("pl_rade", 1.0, 0.57))      # radius, Earth=1, weight
    if "P_DENSITY" in df.columns:
        params.append(("P_DENSITY", 1.0, 1.07))
    if "P_ESCAPE" in df.columns:
        params.append(("P_ESCAPE", 1.0, 0.70))
    if "pl_eqt" in df.columns:
        params.append(("pl_eqt", 255.0, 5.58))      # equilibrium temp proxy

    if len(params) < 2:
        return

    n = len(params)
    esi = pd.Series(1.0, index=df.index)
    valid = mask.copy()

    for col, earth_val, weight in params:
        x = df[col]
        valid &= x.notna() & (x + earth_val > 0)
        similarity = (1.0 - np.abs(x - earth_val) / (x + earth_val)) ** (weight / n)
        esi *= similarity

    df.loc[valid, "P_ESI"] = esi[valid]
    filled = valid.sum()
    if filled > 0:
        log.info(f"Computed ESI for {filled} planets")


def _classify_planet_type(df: pd.DataFrame) -> None:
    """Classify planets by mass/radius into standard categories."""
    if "P_TYPE" not in df.columns:
        df["P_TYPE"] = np.nan

    mask = df["P_TYPE"].isna()

    if "pl_rade" in df.columns:
        r = df["pl_rade"]
        conditions = [
            mask & (r <= 0.5),
            mask & (r > 0.5) & (r <= 1.5),
            mask & (r > 1.5) & (r <= 2.5),
            mask & (r > 2.5) & (r <= 6.0),
            mask & (r > 6.0),
        ]
        choices = ["Miniterran", "Terran", "Superterran", "Neptunian", "Jovian"]
        df["P_TYPE"] = np.select(conditions, choices, default=df["P_TYPE"])
        filled = sum(c.sum() for c in conditions)
        if filled > 0:
            log.info(f"Classified planet type for {filled} planets")


# ---------------------------------------------------------------------------
# Step 6: Final assembly
# ---------------------------------------------------------------------------
def assemble_final(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up and produce the final consolidated DataFrame."""
    df = df.copy()

    # Add provenance flags
    if "source_gaia" not in df.columns:
        df["source_gaia"] = False
    if "source_simbad" not in df.columns:
        df["source_simbad"] = False

    df["source_hec"] = df.get("P_HABITABLE", pd.Series(dtype=float)).notna() | df.get("P_ESI", pd.Series(dtype=float)).notna()
    df["habitability_labeled"] = df.get("P_HABITABLE", pd.Series(dtype=float)).notna()

    # Drop metadata columns
    cols_to_drop = []
    for pattern in DROP_COLS_PATTERNS:
        cols_to_drop.extend([c for c in df.columns if pattern in c.lower()])
    # Also drop internal columns
    cols_to_drop.extend([c for c in df.columns if c.startswith("_")])

    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the consolidated dataset."""
    print("\n" + "=" * 70)
    print("CONSOLIDATED DATASET SUMMARY")
    print("=" * 70)
    print(f"Total planets:     {len(df)}")
    print(f"Total features:    {len(df.columns)}")
    print()

    # Source breakdown
    print("Source coverage:")
    print(f"  NASA (base):     {len(df)} (100%)")
    if "source_hec" in df.columns:
        n = df["source_hec"].sum()
        print(f"  HEC matched:     {n} ({n/len(df)*100:.1f}%)")
    if "source_gaia" in df.columns:
        n = df["source_gaia"].sum()
        print(f"  Gaia matched:    {n} ({n/len(df)*100:.1f}%)")
    if "source_simbad" in df.columns:
        n = df["source_simbad"].sum()
        print(f"  SIMBAD matched:  {n} ({n/len(df)*100:.1f}%)")
    print()

    # Habitability label distribution
    if "P_HABITABLE" in df.columns:
        print("Habitability labels:")
        labeled = df["P_HABITABLE"].notna().sum()
        print(f"  Labeled:         {labeled}")
        print(f"  Unlabeled (NaN): {len(df) - labeled}")
        if labeled > 0:
            print(f"  Distribution:\n{df['P_HABITABLE'].value_counts().to_string(header=False)}")
    print()

    # Key feature completeness
    key_features = [
        "pl_bmasse", "pl_rade", "pl_orbper", "pl_orbsmax", "pl_orbeccen",
        "pl_eqt", "pl_insol", "pl_dens",
        "st_teff", "st_lum", "st_mass", "st_rad", "st_met", "st_logg",
        "st_spectype", "st_age",
        "P_ESI", "P_GRAVITY", "P_DENSITY", "P_ESCAPE",
    ]
    print("Key feature completeness:")
    for feat in key_features:
        if feat in df.columns:
            n = df[feat].notna().sum()
            pct = n / len(df) * 100
            print(f"  {feat:20s}  {n:5d} / {len(df)}  ({pct:5.1f}%)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidate exoplanet data from multiple sources.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-download all data even if cached.")
    args = parser.parse_args()
    force = args.force_refresh

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: NASA base table
    log.info("=" * 50)
    log.info("STEP 1: Downloading NASA Exoplanet Archive PS table")
    log.info("=" * 50)
    nasa = download_nasa_ps(force=force)

    # Step 2: HEC habitability labels
    log.info("=" * 50)
    log.info("STEP 2: Loading & merging HEC habitability data")
    log.info("=" * 50)
    hec = load_hec()
    df = merge_hec(nasa, hec)

    # Step 3: Gaia stellar parameters
    log.info("=" * 50)
    log.info("STEP 3: Querying Gaia DR3 for stellar parameters")
    log.info("=" * 50)
    gaia_df = query_gaia(nasa, force=force)
    df = merge_gaia(df, gaia_df)

    # Step 4: SIMBAD spectral types
    log.info("=" * 50)
    log.info("STEP 4: Querying SIMBAD for spectral types")
    log.info("=" * 50)
    hostnames = df["hostname"].dropna().unique().tolist() if "hostname" in df.columns else []
    simbad_df = query_simbad(hostnames, force=force)
    df = merge_simbad(df, simbad_df)

    # Step 5: Derived features
    log.info("=" * 50)
    log.info("STEP 5: Computing derived features")
    log.info("=" * 50)
    df = compute_derived_features(df)

    # Step 6: Final assembly
    log.info("=" * 50)
    log.info("STEP 6: Final assembly")
    log.info("=" * 50)
    df = assemble_final(df)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Saved consolidated dataset to {OUTPUT_FILE}")

    print_summary(df)


if __name__ == "__main__":
    main()
