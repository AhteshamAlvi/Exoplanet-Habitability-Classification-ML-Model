# Exoplanet Habitability Classification Using Machine Learning

A machine learning project for classifying exoplanet habitability by consolidating data from four major astronomical catalogs into a single, feature-rich dataset. The pipeline merges the NASA Exoplanet Archive, the Habitable Exoplanets Catalog (HEC), ESA's Gaia DR3, and SIMBAD to maximize both the number of features and datapoints available for model training.

The classification task follows a semi-supervised approach: the model trains on the ~5,600 planets with known habitability labels from HEC, then predicts habitability for the ~600 unlabeled confirmed exoplanets from NASA. Planets are classified into three habitability classes: **non-habitable**, **mesoplanet** (Earth-like surface temperatures of 0--50 C), and **psychroplanet** (surface temperatures of -50--0 C).

---

## Data Sources

### 1. NASA Exoplanet Archive -- Planetary Systems (PS) Table

The primary base dataset. Contains all confirmed exoplanets with comprehensive physical, orbital, and stellar parameters. The archive is maintained by the NASA Exoplanet Science Institute (NExScI) at Caltech under contract with NASA.

- **URL**: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS
- **Access method**: TAP (Table Access Protocol) synchronous query
- **Query**: `SELECT * FROM ps WHERE default_flag = 1` -- selects one canonical (best-vetted) row per planet
- **Yield**: ~6,153 confirmed planets x 355 columns
- **Key columns**: planet mass, radius, orbital period, semi-major axis, eccentricity, equilibrium temperature, insolation flux, stellar effective temperature, luminosity, mass, radius, metallicity, surface gravity, spectral type, age, RA/DEC, Gaia DR3 source ID, discovery method/year/facility

### 2. Habitable Exoplanets Catalog (HEC)

Provides habitability labels, derived physical properties, and habitable zone classifications. Maintained by the Planetary Habitability Laboratory (PHL) at the University of Puerto Rico at Arecibo. This is the same PHL-EC dataset used by Basak et al. (2021) for their ML classification experiments.

- **File**: `data/hwc.csv` (pre-downloaded)
- **Source**: PHL Exoplanets Catalog, http://phl.upr.edu/projects/habitable-exoplanets-catalog
- **Yield**: 5,599 planets x 118 columns
- **Key columns**: `P_HABITABLE` (habitability class), `P_ESI` (Earth Similarity Index), `P_TYPE` (planet type classification), habitable zone flags (optimistic/conservative), derived properties (gravity, density, escape velocity, Hill sphere, stellar flux, equilibrium/surface temperature estimates), habitable zone boundaries, snow line, tidal lock indicators

### 3. ESA Gaia DR3

Fills missing stellar parameters using precision astrometry and astrophysical characterization from the Gaia space mission. Queried via two tables: `gaiadr3.gaia_source` for core stellar parameters and `gaiadr3.astrophysical_parameters` for FLAME pipeline outputs (luminosity, mass, radius, age).

- **URL**: https://gea.esac.esa.int/archive/
- **Access method**: TAP synchronous query, batched in groups of 500 source IDs
- **Cross-match**: Linked via `gaia_dr3_id` column in NASA PS data (exact match by Gaia DR3 source ID)
- **Yield**: 4,239 stellar records
- **Key columns**: `teff_gspphot` (effective temperature), `logg_gspphot` (surface gravity), `mh_gspphot` (metallicity), `lum_flame` (luminosity), `mass_flame`, `radius_flame`, `age_flame`, `radial_velocity`, `parallax`
- **Reference**: Gaia Collaboration et al. (2023). "Gaia Data Release 3: Summary of the content and survey properties." *Astronomy & Astrophysics*, 674, A1. DOI: [10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)

### 4. SIMBAD Astronomical Database

Fills remaining gaps, primarily spectral type classifications. SIMBAD is the reference database for astronomical object identification, maintained by the Centre de Donnees astronomiques de Strasbourg (CDS).

- **URL**: https://simbad.u-strasbg.fr/simbad/
- **Access method**: TAP synchronous query via the `ident` and `basic` tables, batched in groups of 200 host star names
- **Cross-match**: Linked via host star name (`hostname` in NASA PS) with SIMBAD alias resolution
- **Yield**: 4,226 stellar records
- **Key columns**: `sp_type` (spectral type), `rvz_radvel` (radial velocity)
- **Reference**: Wenger et al. (2000). "The SIMBAD astronomical database." *Astronomy & Astrophysics Supplement Series*, 143(1), 9-22. DOI: [10.1051/aas:2000332](https://doi.org/10.1051/aas:2000332)

---

## Pipeline Overview

`data_consolidation.py` executes a six-step pipeline:

1. **Download NASA PS Table** -- TAP query for all confirmed planets with `default_flag=1`, cached to `data/nasa_ps.csv`
2. **Load & Merge HEC** -- Left-join habitability labels and derived properties onto NASA base using normalized planet names (90.5% match rate)
3. **Query Gaia DR3** -- Batch TAP queries for stellar parameters using Gaia source IDs from NASA data, with gap-filling for `st_teff`, `st_lum`, `st_mass`, `st_rad`, `st_logg`, `st_met`, `st_age`
4. **Query SIMBAD** -- Batch TAP queries for spectral types using host star names with alias resolution
5. **Compute Derived Features** -- For planets not in HEC: surface gravity, density, escape velocity, equilibrium temperature, habitable zone boundaries (Kopparapu et al. 2013), Earth Similarity Index (ESI), and planet type classification
6. **Final Assembly** -- Merge all sources, add provenance flags, save to `data/consolidated_exoplanets.csv`

---

## Results

`data_consolidation.py` produces `data/consolidated_exoplanets.csv` with:

- **6,153 planets** (all confirmed from NASA) x **398 features**
- **90.5% HEC match rate** -- 5,569 planets have habitability labels
- **82.5% Gaia match rate** -- 8,886 missing stellar values filled
- **46.8% SIMBAD match rate** -- 2,168 spectral types filled

### Key improvements from multi-source fusion

| Feature | NASA + HEC | + Gaia | + SIMBAD |
|---|---|---|---|
| `st_teff` (effective temperature) | 88.0% | **94.0%** | 94.0% |
| `st_lum` (stellar luminosity) | 23.9% | **87.0%** | 87.0% |
| `st_mass` (stellar mass) | 87.6% | **96.5%** | 96.5% |
| `st_rad` (stellar radius) | 86.9% | **93.4%** | 93.4% |
| `st_logg` (surface gravity) | 82.3% | **92.8%** | 92.8% |
| `st_met` (metallicity) | 70.6% | **91.6%** | 91.6% |
| `st_age` (stellar age) | 48.7% | **77.2%** | 77.2% |
| `st_spectype` (spectral type) | 22.0% | 22.0% | **57.2%** |
| `pl_eqt` (equilibrium temperature) | 27.6% | 27.6% | **59.3%** |

Note: `pl_eqt` improves in the final column because additional `st_lum` values from Gaia enable the derived equilibrium temperature computation in Step 5, which runs after all merges.


### Habitability label distribution

| Class | Count | Description |
|---|---|---|
| Non-habitable (0) | 5,499 | Planets without thermal properties to sustain life |
| Psychroplanet (2) | 41 | Mean surface temperature between -50 C and 0 C |
| Mesoplanet (1) | 29 | Mean surface temperature between 0 C and 50 C (Earth-like) |
| Unlabeled (NaN) | 584 | To be predicted by the trained ML model |

---

## Usage

```bash
python data_consolidation.py              # uses caches
python data_consolidation.py --force-refresh  # re-downloads everything
```

On first run, the script downloads from all remote sources (~2 minutes). Subsequent runs use cached intermediate files (`data/nasa_ps.csv`, `data/gaia_stellar.csv`, `data/simbad_stellar.csv`) and complete in seconds.

### Requirements

```
pandas
numpy
requests
astroquery
```

---

## Project Structure

```
exoplanet_ML_classification/
  data_consolidation.py          # Multi-source data pipeline
  ml_build.ipynb                 # ML model notebook (in development)
  data/
    hwc.csv                      # HEC source data (pre-downloaded)
    nasa_ps.csv                  # NASA PS cache (generated)
    gaia_stellar.csv             # Gaia DR3 cache (generated)
    simbad_stellar.csv           # SIMBAD cache (generated)
    consolidated_exoplanets.csv  # Final consolidated dataset (generated)
```

---

## Derived Features

For planets not covered by HEC, the pipeline computes the following derived features:

- **Surface gravity** (Earth units): $g = M / R^2$ where $M$ and $R$ are in Earth masses and radii
- **Density** (Earth units): $\rho = M / R^3$
- **Escape velocity** (Earth units): $v_{esc} = \sqrt{M / R}$
- **Equilibrium temperature** (K): $T_{eq} = T_\star \sqrt{R_\star / 2a}$ where $a$ is the semi-major axis
- **Habitable zone boundaries** (AU): Optimistic and conservative inner/outer edges using the Kopparapu et al. (2013) parameterization based on stellar effective temperature and luminosity
- **Habitable zone flag**: Whether the planet's semi-major axis falls within the optimistic HZ
- **Earth Similarity Index (ESI)**: Geometric mean of similarity in radius, density, escape velocity, and equilibrium temperature relative to Earth
- **Planet type classification**: Miniterran, Terran, Superterran, Neptunian, or Jovian based on radius thresholds

---

## References

Basak, S., Mathur, A., Theophilus, A. J., Deshpande, G., & Murthy, J. (2021). Habitability classification of exoplanets: a machine learning insight. *The European Physical Journal Special Topics*, 230, 2221-2251. DOI: [10.1140/epjs/s11734-021-00203-z](https://doi.org/10.1140/epjs/s11734-021-00203-z)

Cockell, C. S., Bush, T., Bryce, C., Direito, S., Fox-Powell, M., Harrison, J. P., ... & Zorzano, M. P. (2016). Habitability: A Review. *Astrobiology*, 16(1), 89-117. DOI: [10.1089/ast.2015.1295](https://doi.org/10.1089/ast.2015.1295)

Gaidos, E., Deschenes, B., Dundon, L., Fagan, K., McNaughton, C., Menviel-Hessler, L., Moskovitz, N., & Workman, M. (2005). Beyond the Principle of Plentitude: A Review of Terrestrial Planet Habitability. *Astrobiology*, 5(2), 100-126. DOI: [10.1089/ast.2005.5.100](https://doi.org/10.1089/ast.2005.5.100)

Lammer, H., Bredehoft, J. H., Coustenis, A., Khodachenko, M. L., Kaltenegger, L., Grasset, O., ... & Rauer, H. (2009). What makes a planet habitable? *Astronomy and Astrophysics Review*, 17, 181-249. DOI: [10.1007/s00159-009-0019-z](https://doi.org/10.1007/s00159-009-0019-z)

Kopparapu, R. K., Ramirez, R., Kasting, J. F., Eymet, V., Robinson, T. D., Mahadevan, S., ... & Deshpande, R. (2013). Habitable zones around main-sequence stars: new estimates. *The Astrophysical Journal*, 765(2), 131. DOI: [10.1088/0004-637X/765/2/131](https://doi.org/10.1088/0004-637X/765/2/131)

Gaia Collaboration, Vallenari, A., Brown, A. G. A., Prusti, T., et al. (2023). Gaia Data Release 3: Summary of the content and survey properties. *Astronomy & Astrophysics*, 674, A1. DOI: [10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)

Wenger, M., Ochsenbein, F., Egret, D., Dubois, P., Bonnarel, F., Borde, S., ... & Monier, R. (2000). The SIMBAD astronomical database. *Astronomy & Astrophysics Supplement Series*, 143(1), 9-22. DOI: [10.1051/aas:2000332](https://doi.org/10.1051/aas:2000332)

NASA Exoplanet Archive. Planetary Systems Table. NASA Exoplanet Science Institute, California Institute of Technology. https://exoplanetarchive.ipac.caltech.edu/

Mendez, A. (2018). PHL's Exoplanet Catalog. Planetary Habitability Laboratory, University of Puerto Rico at Arecibo. http://phl.upr.edu/projects/habitable-exoplanets-catalog
