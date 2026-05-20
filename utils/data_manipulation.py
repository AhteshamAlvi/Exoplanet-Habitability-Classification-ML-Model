"""
data_manipulation.py — Exoplanet ML Classification
Loads consolidated_exoplanets.csv, cleans, imputes, splits, and balances.
Call the three step functions directly from model notebooks so model-specific
preprocessing can be inserted between imputation and splitting.
"""
from .dependencies import *

# ── Constants ──
DATA_PATH = Path(__file__).parent.parent / "data" / "consolidated_exoplanets.csv"
TARGET = "P_HABITABLE"
RANDOM_STATE = 42

PLANET_FEATURES = [
    "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_bmasse",
    "pl_rade", "pl_dens", "pl_eqt", "pl_insol",
]
STELLAR_FEATURES = [
    "st_teff", "st_rad", "st_mass", "st_lum",
    "st_met", "st_logg", "st_age",
]
DERIVED_FEATURES = ["P_GRAVITY", "P_DENSITY", "P_ESCAPE"]
SYSTEM_FEATURES = ["sy_dist"]

BASEFEATURES = PLANET_FEATURES + STELLAR_FEATURES + DERIVED_FEATURES + SYSTEM_FEATURES

# After Tier 1 drops (pl_dens 83% missing, pl_insol 88% missing)
FEATURES = [
    "pl_orbper", "pl_orbsmax", "pl_orbeccen", "pl_bmasse", "pl_rade",
    "pl_eqt",
    "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "st_logg", "st_age",
    "P_GRAVITY", "P_DENSITY", "P_ESCAPE",
    "sy_dist",
]


def load_and_filter(verbose=True):
    """Load CSV, keep relevant columns, drop rows without target."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[BASEFEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])

    # Tier 1: drop high-missingness features
    df = df[FEATURES + [TARGET]].copy()
    df = df.dropna(subset=[TARGET])

    if verbose:
        print(f"Loaded: {df.shape}")
        print(f"Nulls before imputation:\n{df[FEATURES].isnull().sum()}\n")
    return df


def impute(df, verbose=True):
    """Multi-tier imputation pipeline."""

    # ── Tier 2: Physics-based imputation ──
    # 2a) Eccentricity → 0.0
    n_ecc = df["pl_orbeccen"].isna().sum()
    df["pl_orbeccen"] = df["pl_orbeccen"].fillna(0.0)
    if verbose:
        print(f"[Tier 2a] pl_orbeccen: filled {n_ecc} with 0.0")

    # 2b) Semi-major axis from Kepler's 3rd law
    mask_a = df["pl_orbsmax"].isna() & df["pl_orbper"].notna() & df["st_mass"].notna()
    P_yr = df.loc[mask_a, "pl_orbper"] / 365.25
    df.loc[mask_a, "pl_orbsmax"] = (P_yr**2 * df.loc[mask_a, "st_mass"]) ** (1 / 3)
    if verbose:
        print(f"[Tier 2b] pl_orbsmax: filled {mask_a.sum()} via Kepler's 3rd law")

    # 2c) Orbital period from Kepler's 3rd law inverse
    mask_p = df["pl_orbper"].isna() & df["pl_orbsmax"].notna() & df["st_mass"].notna()
    a = df.loc[mask_p, "pl_orbsmax"]
    P_yr_inv = np.sqrt(a**3 / df.loc[mask_p, "st_mass"])
    df.loc[mask_p, "pl_orbper"] = P_yr_inv * 365.25
    if verbose:
        print(f"[Tier 2c] pl_orbper: filled {mask_p.sum()} via Kepler's 3rd law (inverse)")

    # 2d) Luminosity from Stefan-Boltzmann
    mask_lum = df["st_lum"].isna() & df["st_teff"].notna() & df["st_rad"].notna()
    L_linear = df.loc[mask_lum, "st_rad"] ** 2 * (df.loc[mask_lum, "st_teff"] / 5778) ** 4
    df.loc[mask_lum, "st_lum"] = np.log10(L_linear)
    if verbose:
        print(f"[Tier 2d] st_lum: filled {mask_lum.sum()} via Stefan-Boltzmann law")

    # 2e) Planet mass from radius via Chen & Kipping (2017)
    mask_mass = df["pl_bmasse"].isna() & df["pl_rade"].notna()
    r = df.loc[mask_mass, "pl_rade"]
    mass_est = np.where(r <= 1.5, r**2.06, 2.7 * r**1.7)
    df.loc[mask_mass, "pl_bmasse"] = mass_est
    if verbose:
        print(f"[Tier 2e] pl_bmasse: filled {mask_mass.sum()} via mass-radius relation")

    # 2f) Planet radius from mass via inverse mass-radius
    mask_rad = df["pl_rade"].isna() & df["pl_bmasse"].notna()
    m = df.loc[mask_rad, "pl_bmasse"]
    rad_est = np.where(m <= 2.0, m**0.485, (m / 2.7) ** 0.588)
    df.loc[mask_rad, "pl_rade"] = rad_est
    if verbose:
        print(f"[Tier 2f] pl_rade: filled {mask_rad.sum()} via inverse mass-radius relation")

    # 2g) Equilibrium temperature
    R_SUN_AU = 6.957e8 / 1.496e11
    mask_eqt = df["pl_eqt"].isna()
    t_eq_calc = df["st_teff"] * np.sqrt(df["st_rad"] * R_SUN_AU / (2 * df["pl_orbsmax"]))
    fillable_eqt = mask_eqt & t_eq_calc.notna()
    df.loc[fillable_eqt, "pl_eqt"] = t_eq_calc[fillable_eqt]
    if verbose:
        print(f"[Tier 2g] pl_eqt: filled {fillable_eqt.sum()} via equilibrium temp formula")

    # 2h-j) Derived features
    mask_derived = df["P_GRAVITY"].isna() & df["pl_bmasse"].notna() & df["pl_rade"].notna()
    df.loc[mask_derived, "P_GRAVITY"] = df.loc[mask_derived, "pl_bmasse"] / df.loc[mask_derived, "pl_rade"] ** 2
    df.loc[mask_derived, "P_DENSITY"] = df.loc[mask_derived, "pl_bmasse"] / df.loc[mask_derived, "pl_rade"] ** 3
    df.loc[mask_derived, "P_ESCAPE"] = np.sqrt(df.loc[mask_derived, "pl_bmasse"] / df.loc[mask_derived, "pl_rade"])
    if verbose:
        print(f"[Tier 2h-j] P_GRAVITY/DENSITY/ESCAPE: filled {mask_derived.sum()} via physics formulas")
        print(f"\nNulls after Tier 2:\n{df[FEATURES].isnull().sum()}\n")

    # ── Tier 3: BayesianRidge MICE for stellar features ──
    stellar_cols = ["st_teff", "st_rad", "st_mass", "st_logg", "st_lum"]
    pre_nulls_stellar = df[stellar_cols].isnull().sum().sum()
    imputer_stellar = IterativeImputer(
        estimator=BayesianRidge(), max_iter=20, random_state=RANDOM_STATE
    )
    df[stellar_cols] = imputer_stellar.fit_transform(df[stellar_cols])
    if verbose:
        post = df[stellar_cols].isnull().sum().sum()
        print(f"[Tier 3] BayesianRidge MICE on stellar features: {pre_nulls_stellar} → {post} nulls")

    # ── Tier 4: RF Iterative Imputer for planetary features ──
    planet_cols = [
        "pl_orbper", "pl_orbsmax", "pl_bmasse", "pl_rade", "pl_eqt",
        "P_GRAVITY", "P_DENSITY", "P_ESCAPE",
    ]
    pre_nulls_planet = df[planet_cols].isnull().sum().sum()
    imputer_planet = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        max_iter=10, random_state=RANDOM_STATE,
    )
    df[planet_cols] = imputer_planet.fit_transform(df[planet_cols])
    if verbose:
        post = df[planet_cols].isnull().sum().sum()
        print(f"[Tier 4] RF Iterative Imputer on planetary features: {pre_nulls_planet} → {post} nulls")

    # ── Tier 5: Feature-specific methods ──
    # 5a) st_met → grouped median by spectral type
    teff_bins = [0, 3700, 5200, 6000, 7500, np.inf]
    teff_labels = ["M", "K", "G", "F", "A+"]
    df["_teff_bin"] = pd.cut(df["st_teff"], bins=teff_bins, labels=teff_labels)
    pre_met = df["st_met"].isna().sum()
    group_medians = df.groupby("_teff_bin", observed=True)["st_met"].transform("median")
    mask_met = df["st_met"].isna()
    df.loc[mask_met, "st_met"] = group_medians[mask_met]
    df["st_met"] = df["st_met"].fillna(df["st_met"].median())
    if verbose:
        print(f"[Tier 5a] st_met: filled {pre_met} via grouped median by spectral type")

    # 5b) st_age → RF regression
    pre_age = df["st_age"].isna().sum()
    mask_age = df["st_age"].isna()
    age_predictors = ["st_teff", "st_rad", "st_mass", "st_lum", "st_met", "st_logg", "sy_dist"]
    train_age = df[~mask_age].dropna(subset=age_predictors)
    if len(train_age) > 50:
        rf_age = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf_age.fit(train_age[age_predictors], train_age["st_age"])
        predict_mask = mask_age & df[age_predictors].notna().all(axis=1)
        df.loc[predict_mask, "st_age"] = rf_age.predict(df.loc[predict_mask, age_predictors])
    df["st_age"] = df["st_age"].fillna(df["st_age"].median())
    if verbose:
        print(f"[Tier 5b] st_age: filled {pre_age} via RF regression + median fallback")

    df.drop(columns=["_teff_bin"], inplace=True)

    # ── Tier 6: KNN for residuals ──
    pre_nulls_total = df[FEATURES].isnull().sum().sum()
    if pre_nulls_total > 0:
        sc = StandardScaler()
        scaled = sc.fit_transform(df[FEATURES])
        imputer_knn = KNNImputer(n_neighbors=5, weights="distance")
        imputed = imputer_knn.fit_transform(scaled)
        df[FEATURES] = sc.inverse_transform(imputed)
        if verbose:
            post = df[FEATURES].isnull().sum().sum()
            print(f"[Tier 6] KNN imputer: {pre_nulls_total} → {post} nulls")

    # ── Tier 7: Global median fallback ──
    remaining = df[FEATURES].isnull().sum().sum()
    if remaining > 0:
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
        if verbose:
            print(f"[Tier 7] Median fallback: filled {remaining} remaining nulls")

    return df


def split_and_balance(df, features=None, scaler=None, smote_variant="standard",
                      test_size=0.2, majority_target=300, minority_target=200, verbose=True):
    """Split, scale, and balance the dataset.

    Parameters
    ----------
    features : list, optional
        Feature columns to use. Defaults to the global FEATURES list.
    scaler : sklearn scaler instance, optional
        Defaults to StandardScaler(). Pass RobustScaler() for SVM.
    smote_variant : {"standard", "borderline"}
        Which SMOTE variant to use for minority oversampling.
    """
    cols = features if features is not None else FEATURES
    scaler_obj = scaler if scaler is not None else StandardScaler()

    X = df[cols].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    if verbose:
        print(f"\nTrain shape: {X_train.shape}")
        print(f"Test shape:  {X_test.shape}")
        print(f"\nTrain distribution (BEFORE balancing):")
        print(y_train.value_counts().sort_index())

    # Scale
    X_train_scaled = pd.DataFrame(
        scaler_obj.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler_obj.transform(X_test), columns=cols, index=X_test.index
    )

    # Balance
    min_class_count = y_train.value_counts().min()
    k_neighbors = min(5, min_class_count - 1)

    smote_cls = BorderlineSMOTE if smote_variant == "borderline" else SMOTE
    sampler = ImbPipeline([
        ("under", RandomUnderSampler(
            sampling_strategy={0.0: majority_target}, random_state=RANDOM_STATE
        )),
        ("smote", smote_cls(
            sampling_strategy={1.0: minority_target, 2.0: minority_target},
            k_neighbors=k_neighbors, random_state=RANDOM_STATE,
        )),
        ("tomek", TomekLinks()),
    ])

    X_train_bal, y_train_bal = sampler.fit_resample(X_train_scaled, y_train)

    # Class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_bal), y=y_train_bal
    )
    class_weight_dict = dict(zip(np.unique(y_train_bal).astype(int), class_weights))

    if verbose:
        print(f"\nTrain distribution (AFTER balancing):")
        print(pd.Series(y_train_bal).value_counts().sort_index())
        print(f"Balanced train shape: {X_train_bal.shape}")
        print(f"Class weights: {class_weight_dict}")
        print(f"\n{'='*60}")
        print("Outputs ready for modeling:")
        print(f"  X_train_bal       → {X_train_bal.shape} (balanced, scaled)")
        print(f"  y_train_bal       → {len(y_train_bal)} samples")
        print(f"  X_test_scaled     → {X_test_scaled.shape} (held-out, scaled)")
        print(f"  y_test            → {len(y_test)} samples (original distribution)")
        print(f"  class_weight_dict → for class_weight param in classifiers")

    return X_train_bal, y_train_bal, X_test_scaled, y_test, class_weight_dict, scaler_obj


if __name__ == "__main__":
    df = load_and_filter(verbose=True)
    df = impute(df, verbose=True)
    X_train_bal, y_train_bal, X_test_scaled, y_test, class_weight_dict, scaler = (
        split_and_balance(df, verbose=True)
    )
    print("\nPipeline complete. All outputs verified.")
