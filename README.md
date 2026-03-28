# Exoplanet Habitability Classification Using Machine Learning

A machine learning project for classifying exoplanet habitability by consolidating data from four major astronomical catalogs into a single, feature-rich dataset. The pipeline merges the NASA Exoplanet Archive, the Habitable Exoplanets Catalog (HEC), ESA's Gaia DR3, and SIMBAD to maximize both the number of features and datapoints available for model training.

The classification task follows a semi-supervised approach: the model trains on the ~5,600 planets with known habitability labels from HEC, then predicts habitability for the ~600 unlabeled confirmed exoplanets from NASA. Planets are classified into three habitability classes: **non-habitable**, **mesoplanet** (Earth-like surface temperatures of 0--50 C), and **psychroplanet** (surface temperatures of -50--0 C).

---

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

---

## Data Cleaning & Imputation

After consolidation, the dataset contains 408 columns across 6,153 planets. Before model training, the ML notebook (`ml_build.ipynb`) performs a multi-stage cleaning and imputation pipeline. The choice of imputation method matters significantly: as Alam et al. (2023) demonstrate, the technique used to handle missing values directly impacts both clustering quality and downstream classification accuracy, with decision tree-based and *k*-NN methods consistently outperforming naive statistical fills across multiple benchmark datasets.

### Step 1 -- Feature Selection

The 408 raw columns are reduced to 17 ML-relevant features. Columns are dropped for three reasons:

- **Identifiers and metadata**: planet/star names, reference strings, discovery provenance, coordinate strings, flag columns -- these carry no predictive signal for habitability.
- **Error and string columns**: measurement uncertainties (`*err1`, `*err2`), limit flags (`*lim`), and string representations (`*str`) are auxiliary to the measurements themselves.

The surviving 19 features span three domains:

| Domain | Features |
|--------|----------|
| Planetary | `pl_orbper`, `pl_orbsmax`, `pl_orbeccen`, `pl_bmasse`, `pl_rade`, `pl_eqt`, `pl_dens`, `pl_insol`|
| Stellar | `st_teff`, `st_rad`, `st_mass`, `st_lum`, `st_met`, `st_logg`, `st_age` |
| Derived | `P_GRAVITY`, `P_DENSITY`, `P_ESCAPE` |
| System | `sy_dist` |

Features that are direct habitability proxies (`P_ESI`, `in_habitable_zone`) are explicitly excluded to prevent data leakage.

### Step 2 -- Target Filtering

Only planets with a known habitability label (`P_HABITABLE` not null) are retained, reducing the dataset from 6,153 to 5,569 rows. The 584 unlabeled planets are set aside for prediction after model training.

### Step 3 -- Tiered Imputation

Missingness across the 17 features ranges from 0.1% to 59.5%. Only 4.9% of rows are fully complete; strict case deletion would leave just 2 habitable planets. A tiered imputation strategy applies the most appropriate method per feature based on its missingness rate, physical meaning, and distributional properties. As Alam et al. (2023) note, "the chosen imputation technique should aim to improve data analysis and deliver unbiased results," and different datasets benefit from different methods -- no single approach is universally optimal.

#### Tier 1 -- Redundant or Excessively Sparse Features

Data which is excessively sparse is dropped. `pl_dens` (82.8% missing) is dropped because the HEC-derived `P_DENSITY` covers the same quantity at 0.1% missingness. `pl_insol` (87.7% missing) is dropped because insolation flux is already captured by the combination of `pl_eqt`, `st_lum`, and `pl_orbsmax`.

#### Tier 2 -- Physics-Based Imputation

Where astrophysical relationships between features are well-established, missing values are computed from other available columns rather than statistically estimated. The order of operations matters because each step unlocks fills for subsequent steps.

| Feature | Formula | Basis |
|---------|---------|-------|
| `pl_orbeccen` (59.5%) | Fill with 0.0 | Unmeasured eccentricities default to circular; most transiting planets have near-zero eccentricity |
| `pl_orbsmax` (40.2%) | $a = (P^2 \times M_\star)^{1/3}$ | Kepler's third law, with period $P$ in years and stellar mass $M_\star$ in solar masses |
| `pl_orbper` (4.7%) | $P = \sqrt{a^3 / M_\star}$ | Kepler's third law (inverse), applied to planets with known semi-major axis but missing period |
| `st_lum` (12.1%) | $\log_{10}(R_\star^2 \times (T_\star / 5778)^4)$ | Stefan-Boltzmann law relating luminosity to stellar radius and effective temperature |
| `pl_bmasse` (53.6%) | $R \leq 1.5 R_\oplus: M = R^{2.06}$; $R > 1.5 R_\oplus: M = 2.7 R^{1.7}$ | Chen & Kipping (2017) empirical mass-radius relation, piecewise by planet type |
| `pl_rade` (24.7%) | $M \leq 2 M_\oplus: R = M^{0.485}$; $M > 2 M_\oplus: R = (M/2.7)^{0.588}$ | Inverse mass-radius relation, applied only to original (non-imputed) masses |
| `pl_eqt` (43.1%) | $T_{eq} = T_\star \sqrt{R_{\star,AU} / 2a}$ | Equilibrium temperature from stellar temperature, radius (in AU), and semi-major axis (now filled from above) |
| `P_GRAVITY` (0.1%) | $M / R^2$ | Surface gravity in Earth units |
| `P_DENSITY` (0.1%) | $M / R^3$ | Density in Earth units |
| `P_ESCAPE` (0.1%) | $\sqrt{M / R}$ | Escape velocity in Earth units |

#### Tier 3 -- Iterative Imputation with Bayesian Ridge (Stellar Features)

The stellar features `st_teff`, `st_rad`, `st_mass`, `st_logg`, and `st_lum` are strongly correlated via HR-diagram physics (e.g., $r = -0.74$ between `st_rad` and `st_logg`). These are imputed using MICE (Multiple Imputation by Chained Equations) with a `BayesianRidge` estimator. MICE models each feature as a regression on the others, iterating until convergence. Bayesian Ridge provides built-in regularization and handles the multicollinearity inherent in stellar parameters. This approach is analogous to the classifier-based imputation methods that Alam et al. (2023) found to produce the highest accuracy and lowest variance compared to simple statistical methods.

#### Tier 4 -- Random Forest Iterative Imputation (Planetary Features)

Remaining gaps in planetary features (`pl_orbper`, `pl_orbsmax`, `pl_bmasse`, `pl_rade`, `pl_eqt`, `P_GRAVITY`, `P_DENSITY`, `P_ESCAPE`) are filled using `IterativeImputer` with a `RandomForestRegressor` estimator. Planetary features exhibit highly nonlinear relationships (e.g., the mass-radius relation is a piecewise power law) and heavy right skew (skewness 2--42). Random Forest captures these nonlinear interactions without assuming normality. Alam et al. (2023) found that decision tree-based imputation consistently achieved the highest accuracy across benchmark datasets, with the lowest average variance from original data -- confirming its suitability for features with complex interdependencies.

#### Tier 5 -- Feature-Specific Methods

Two stellar features require specialized treatment due to their weak correlations with all other features:

- **`st_met` (metallicity, 7.6% missing)** -- Imputed via **grouped median** by spectral type bin (M/K/G/F/A+, binned from `st_teff`). Metallicity has a maximum $|r| = 0.09$ with all other stellar features; regression-based imputation would effectively predict the global mean. Grouped median respects that different spectral types sample different galactic stellar populations. Alam et al. (2023) similarly found that class-based median imputation -- filling within homogeneous subgroups rather than globally -- preserves class-specific distributional characteristics.
- **`st_age` (stellar age, 22.4% missing)** -- Imputed via **Random Forest regression** using all other stellar features plus `sy_dist` as predictors. Age has the weakest correlations of any stellar feature (maximum $|r| = 0.24$ with `st_mass`), making linear methods unreliable. However, the relationship between age and other stellar parameters is nonlinear (e.g., post-main-sequence evolution), which Random Forest captures. Residual nulls (where predictors themselves are missing) fall back to global median.

#### Tier 6 -- *k*-NN Distance-Based Imputation

Any remaining nulls (primarily `sy_dist` at 2.1%) are filled using `KNNImputer` with $k = 5$ neighbors, weighted by inverse distance. Features are standardized before neighbor computation and inverse-transformed after. With low missingness and many complete neighbors available, *k*-NN produces accurate fills by leveraging the local structure of the data. As Alam et al. (2023) describe, "*k*-NN imputation leverages the proximity of instances to ensure that imputed values reflect the local data structure accurately."

#### Tier 7 -- Global Median Fallback

A final global median fill catches any edge-case nulls remaining after all prior tiers. In practice, this tier fills zero or near-zero values.

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

Thus the final 17 features going into modeling are:
| Domain | Features |
|--------|----------|
| Planetary | `pl_orbper`, `pl_orbsmax`, `pl_orbeccen`, `pl_bmasse`, `pl_rade`, `pl_eqt` |
| Stellar | `st_teff`, `st_rad`, `st_mass`, `st_lum`, `st_met`, `st_logg`, `st_age` |
| Derived | `P_GRAVITY`, `P_DENSITY`, `P_ESCAPE` |
| System | `sy_dist` |


## References

Alam, S., Ayub, M. S., Arora, S., & Khan, M. A. (2023). An investigation of the imputation techniques for missing values in ordinal data enhancing clustering and classification analysis validity. *Decision Analytics Journal*, 9, 100341. DOI: [10.1016/j.dajour.2023.100341](https://doi.org/10.1016/j.dajour.2023.100341)

Basak, S., Mathur, A., Theophilus, A. J., Deshpande, G., & Murthy, J. (2021). Habitability classification of exoplanets: a machine learning insight. *The European Physical Journal Special Topics*, 230, 2221-2251. DOI: [10.1140/epjs/s11734-021-00203-z](https://doi.org/10.1140/epjs/s11734-021-00203-z)

Chen, J., & Kipping, D. (2017). Probabilistic forecasting of the masses and radii of other worlds. *The Astrophysical Journal*, 834(1), 17. DOI: [10.3847/1538-4357/834/1/17](https://doi.org/10.3847/1538-4357/834/1/17)

Cockell, C. S., Bush, T., Bryce, C., Direito, S., Fox-Powell, M., Harrison, J. P., ... & Zorzano, M. P. (2016). Habitability: A Review. *Astrobiology*, 16(1), 89-117. DOI: [10.1089/ast.2015.1295](https://doi.org/10.1089/ast.2015.1295)

Gaidos, E., Deschenes, B., Dundon, L., Fagan, K., McNaughton, C., Menviel-Hessler, L., Moskovitz, N., & Workman, M. (2005). Beyond the Principle of Plentitude: A Review of Terrestrial Planet Habitability. *Astrobiology*, 5(2), 100-126. DOI: [10.1089/ast.2005.5.100](https://doi.org/10.1089/ast.2005.5.100)

Lammer, H., Bredehoft, J. H., Coustenis, A., Khodachenko, M. L., Kaltenegger, L., Grasset, O., ... & Rauer, H. (2009). What makes a planet habitable? *Astronomy and Astrophysics Review*, 17, 181-249. DOI: [10.1007/s00159-009-0019-z](https://doi.org/10.1007/s00159-009-0019-z)

Kopparapu, R. K., Ramirez, R., Kasting, J. F., Eymet, V., Robinson, T. D., Mahadevan, S., ... & Deshpande, R. (2013). Habitable zones around main-sequence stars: new estimates. *The Astrophysical Journal*, 765(2), 131. DOI: [10.1088/0004-637X/765/2/131](https://doi.org/10.1088/0004-637X/765/2/131)

Gaia Collaboration, Vallenari, A., Brown, A. G. A., Prusti, T., et al. (2023). Gaia Data Release 3: Summary of the content and survey properties. *Astronomy & Astrophysics*, 674, A1. DOI: [10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)

Wenger, M., Ochsenbein, F., Egret, D., Dubois, P., Bonnarel, F., Borde, S., ... & Monier, R. (2000). The SIMBAD astronomical database. *Astronomy & Astrophysics Supplement Series*, 143(1), 9-22. DOI: [10.1051/aas:2000332](https://doi.org/10.1051/aas:2000332)

NASA Exoplanet Archive. Planetary Systems Table. NASA Exoplanet Science Institute, California Institute of Technology. https://exoplanetarchive.ipac.caltech.edu/

Mendez, A. (2018). PHL's Exoplanet Catalog. Planetary Habitability Laboratory, University of Puerto Rico at Arecibo. http://phl.upr.edu/projects/habitable-exoplanets-catalog
