# Exoplanet Classification Machine Learning Model

### Model for classifying habitable planets based on available data from NASA exoplanet database.

---

## Data Sources & APIs

All data should be pulled via API when possible to keep training data fresh, reproducible, and out of version control.

### Exoplanet-Specific APIs

#### NASA Exoplanet Archive — TAP Service
Programmatic access to the Planetary Systems (PS) table (the source of `exoplanets.csv`). Supports ADQL queries for flexible column selection, joins, and filtering.
- Endpoint: [https://exoplanetarchive.ipac.caltech.edu/TAP](https://exoplanetarchive.ipac.caltech.edu/TAP)
- Docs: [https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)

#### Habitable Exoplanets Catalog (HEC) — UPR Arecibo
Source of `hwc.csv`. Provides Earth Similarity Index (ESI), habitable zone classifications, and `P_HABITABLE` labels used as ground truth.
- Info: [http://phl.upr.edu/projects/habitable-exoplanets-catalog](http://phl.upr.edu/projects/habitable-exoplanets-catalog)

#### Gaia DR3 Archive
High-precision stellar parameters (temperature, luminosity, radius, age, metallicity) for host stars. Cross-match on `gaia_dr3_id` to fill NaN gaps in NASA's stellar columns.
- TAP Endpoint: [https://gea.esac.esa.int/tap-server/tap](https://gea.esac.esa.int/tap-server/tap)
- Docs: [https://gea.esac.esa.int/archive/](https://gea.esac.esa.int/archive/)

#### MAST (Mikulski Archive for Space Telescopes)
Kepler, K2, and TESS data. Useful for light curve-derived properties (transit depth, stellar variability).
- API Docs: [https://mast.stsci.edu/api/v0/](https://mast.stsci.edu/api/v0/)

#### exoplanet.eu (European Exoplanet Encyclopaedia)
Independent catalog with different observational coverage. Good for cross-validation.
- API: [http://exoplanet.eu/API/](http://exoplanet.eu/API/)

#### SIMBAD (CDS Strasbourg)
Stellar cross-references for filling in spectral types and metallicity where NASA's archive has NaN.
- TAP Endpoint: [http://simbad.u-strasbg.fr/simbad/sim-tap](http://simbad.u-strasbg.fr/simbad/sim-tap)

---

### Broader Solar System & Extraterrestrial Body APIs

#### JPL Horizons API
Ephemeris and physical data for virtually every known body in the solar system: 1.4M+ asteroids, 4,000+ comets, 424 planetary satellites, all planets, the Sun, and select spacecraft. Returns orbital elements, state vectors, and observer ephemerides.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/horizons.html](https://ssd-api.jpl.nasa.gov/doc/horizons.html)
- Python access via `astroquery.jplhorizons`

#### JPL Small-Body Database (SBDB) API
Orbital and physical data for all known asteroids and comets. Supports lookups by designation and complex constraint-based queries.
- Lookup: [https://ssd-api.jpl.nasa.gov/doc/sbdb.html](https://ssd-api.jpl.nasa.gov/doc/sbdb.html)
- Query: [https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html)

#### JPL Small-Body Close Approach (CAD) API
Close-approach data for asteroids and comets relative to planets. Includes distance, velocity, and uncertainty data.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/cad.html](https://ssd-api.jpl.nasa.gov/doc/cad.html)

#### JPL Sentry API
NEO Earth impact risk assessment data. Provides impact probabilities and Palermo/Torino scale ratings.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/sentry.html](https://ssd-api.jpl.nasa.gov/doc/sentry.html)

#### JPL Scout API
Orbit, ephemeris, and impact risk data for newly discovered objects on the NEO Confirmation Page.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/scout.html](https://ssd-api.jpl.nasa.gov/doc/scout.html)

#### JPL Fireball API
Atmospheric impact data for bolides/fireballs reported by US Government sensors. Includes energy, velocity, and location.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/fireball.html](https://ssd-api.jpl.nasa.gov/doc/fireball.html)

#### JPL NHATS API
Data on near-Earth asteroids accessible for human exploration missions. Includes delta-v, mission duration, and launch window data.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/nhats.html](https://ssd-api.jpl.nasa.gov/doc/nhats.html)

#### JPL Small-Body Satellites API
Data on satellites (moons) of asteroids and other small bodies.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/sb_sat.html](https://ssd-api.jpl.nasa.gov/doc/sb_sat.html)

#### NASA NeoWs (Near Earth Object Web Service)
RESTful service for near-Earth asteroid information. Search by close-approach date, look up by JPL small body ID, or browse the full dataset. Requires a free NASA API key.
- Endpoint: `https://api.nasa.gov/neo/rest/v1/`
- API Key: [https://api.nasa.gov/](https://api.nasa.gov/)

#### Solar System OpenData API
Community REST API covering all solar system planets, moons, dwarf planets, asteroids, and TNOs. Returns mass, dimensions, gravity, orbital parameters, and discovery info.
- Endpoint: [https://api.le-systeme-solaire.net/rest/bodies/](https://api.le-systeme-solaire.net/rest/bodies/)
- Docs: [https://api.le-systeme-solaire.net/en/](https://api.le-systeme-solaire.net/en/)

---

### Additional JPL SSD APIs (Utility)

#### JD Date/Time Converter
Julian Day number to/from calendar date/time.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/jd_cal.html](https://ssd-api.jpl.nasa.gov/doc/jd_cal.html)

#### Periodic Orbits API
Database of periodic orbits in the circular restricted three-body problem.
- Endpoint: [https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html](https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html)

---

## Tech Stack

- PyTorch
- pandas / numpy
- matplotlib

## Project Status

Early stage — data loading implemented, model not yet built.
