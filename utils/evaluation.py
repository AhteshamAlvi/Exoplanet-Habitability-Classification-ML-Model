"""
evaluation.py — Metrics for the exoplanet ML pipeline.

The headline metrics come from `run_seed_sweep`: per-class F1 mean ± std across
many train/test splits, which `print_sweep_classification_report` formats and
`plot_sweep_f1_bar` (in plots.py) visualises.

The single-split helpers retained here are the ones the new Metrics section
still uses:
  - run_stratified_cv: within-seed CV stability check (orthogonal to the
    across-split variance reported by the sweep)
  - print_error_patterns: concrete off-diagonal report for a single split
    (averaging confusion matrices loses interpretability)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from .constants import get_class_meta
from .data_manipulation import split_and_balance


# ── Single-split diagnostics ──

def run_stratified_cv(model, X_train, y_train, n_splits=5):
    """Stratified k-fold CV scored on f1_macro.

    Measures within-seed tuning stability — orthogonal to the across-split
    variance reported by run_seed_sweep. Prints mean ± std and returns the
    score array.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"\n{n_splits}-Fold CV F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def print_error_patterns(y_test, y_pred):
    """Print every off-diagonal confusion cell as 'TrueClass → PredClass: N (X%)'.

    Reports a single split (seed=42) — averaged error patterns lose the
    instance-level interpretability that makes this diagnostic useful.
    """
    class_names, classes, _ = get_class_meta()
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("Key error patterns:")
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                print(
                    f"  {true_name} → predicted {pred_name}: {cm[i, j]} "
                    f"({cm_norm[i, j]:.1%} of true {true_name})"
                )


# ── Seed-sweep pipeline ──

def run_seed_sweep(model_factory, df, *, n_seeds=50, scaler_cls=None,
                   features=None, smote_variant="standard",
                   model_name="model", verbose=True, save_csv=None):
    """Multi-seed train/test sweep with fixed hyperparameters.

    Fits a fresh model per seed using `model_factory(class_weight_dict)`, then
    records per-class and macro/weighted F1 on each seed's held-out split.

    Parameters
    ----------
    model_factory : callable(class_weight_dict) -> sklearn estimator
        Returns a fresh estimator per seed. Models that don't accept
        class_weight (GNB, KNN, LDA, QDA, MLP, XGB) can ignore the argument.
    df : pandas.DataFrame
        Post-physics-imputation frame. Tier 3-7 imputers are re-fit on each
        seed's training split inside split_and_balance(), so passing the same
        df across the loop is safe — no leakage.
    n_seeds : int
        Number of random_state values to sweep (0 .. n_seeds-1).
    scaler_cls : class, optional
        e.g. StandardScaler (default) or RobustScaler. A fresh instance is
        constructed per seed.
    features : list of str, optional
        Column subset (e.g. GNB drops the three derived columns).
    smote_variant : {"standard", "borderline"}
    save_csv : str or Path, optional
        If provided, write the per-seed DataFrame here.

    Returns
    -------
    pandas.DataFrame
        Columns: seed, f1_macro, f1_weighted, f1_nonhab, f1_meso, f1_psychro.
    """
    if scaler_cls is None:
        scaler_cls = StandardScaler

    rows = []
    for seed in range(n_seeds):
        kwargs = {"scaler": scaler_cls(), "smote_variant": smote_variant,
                  "random_state": seed, "verbose": False}
        if features is not None:
            kwargs["features"] = features

        X_tr, y_tr, X_te, y_te, cw, _ = split_and_balance(df, **kwargs)

        est = model_factory(cw)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)

        f1_per = f1_score(y_te, y_pred, average=None,
                          labels=[0.0, 1.0, 2.0], zero_division=0)
        rows.append({
            "seed": seed,
            "f1_macro":    f1_score(y_te, y_pred, average="macro",    zero_division=0),
            "f1_weighted": f1_score(y_te, y_pred, average="weighted", zero_division=0),
            "f1_nonhab":   f1_per[0],
            "f1_meso":     f1_per[1],
            "f1_psychro":  f1_per[2],
        })

    sweep_df = pd.DataFrame(rows)

    if verbose:
        m, s = sweep_df["f1_macro"].mean(), sweep_df["f1_macro"].std()
        sem = s / np.sqrt(len(sweep_df))
        print(f"{model_name} — {n_seeds}-seed sweep")
        print(f"  F1 macro:    {m:.4f} ± {s:.4f}  (SEM {sem:.4f}, "
              f"min {sweep_df['f1_macro'].min():.4f}, "
              f"max {sweep_df['f1_macro'].max():.4f})")
        print(f"  F1 weighted: {sweep_df['f1_weighted'].mean():.4f} "
              f"± {sweep_df['f1_weighted'].std():.4f}")

    if save_csv is not None:
        sweep_df.to_csv(save_csv, index=False)
        if verbose:
            print(f"  Saved to {save_csv}")

    return sweep_df


def print_sweep_classification_report(sweep_df, model_name):
    """Per-class F1 mean ± std (min, max) across seeds.

    Sweep-aware replacement for the single-seed classification report — gives
    an honest picture of per-class scoring variance when the minority test
    classes are tiny (6 meso, 8 psychro samples).
    """
    class_names, _, _ = get_class_meta()
    col_for = {"Non-Habitable": "f1_nonhab",
               "Mesoplanet":    "f1_meso",
               "Psychroplanet": "f1_psychro"}
    n = len(sweep_df)

    print("=" * 64)
    print(f"{model_name} — Classification Report ({n}-seed sweep)")
    print("=" * 64)
    print(f"{'Class':<16} {'F1 mean':>10} {'F1 std':>10} {'F1 min':>10} {'F1 max':>10}")
    print("-" * 64)
    for name in class_names:
        s = sweep_df[col_for[name]]
        print(f"{name:<16} {s.mean():>10.4f} {s.std():>10.4f} "
              f"{s.min():>10.4f} {s.max():>10.4f}")
    print("-" * 64)
    for label, col in [("Macro avg", "f1_macro"), ("Weighted avg", "f1_weighted")]:
        s = sweep_df[col]
        print(f"{label:<16} {s.mean():>10.4f} {s.std():>10.4f} "
              f"{s.min():>10.4f} {s.max():>10.4f}")
