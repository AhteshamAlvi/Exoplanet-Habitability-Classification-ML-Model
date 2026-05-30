"""
evaluation.py — Metrics for the exoplanet ML pipeline.

`run_seed_sweep` is the workhorse: trains a fresh model on each of N random
train/test splits using fixed hyperparameters, then captures per-seed F1,
precision, recall, accuracy, balanced accuracy, MCC, minority-F1-macro, plus
the confusion matrix, predictions, and (when available) feature importances.

The result is a SweepResult, consumed by:
  - print_sweep_classification_report  : per-class F1 table (this file)
  - print_sweep_performance_summary    : per-class P/R + aggregate robustness
                                         metrics, sized for cross-model judging
  - print_sweep_error_patterns         : off-diagonal CM cells, aggregated
  - plot_sweep_f1_bar                  : per-class bar chart with error bars
  - plot_sweep_confusion_matrix        : averaged normalized CM with per-cell std
  - plot_sweep_feature_importances     : mean ± std importance per feature

Single-split helper kept here:
  - run_stratified_cv : within-seed CV stability check, orthogonal to sweep
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from .constants import get_class_meta
from .data_manipulation import split_and_balance
from .dependencies import SWEEP_DIR


# ── Single-split diagnostic kept from before ──

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


# ── Seed-sweep pipeline ──

@dataclass
class SweepResult:
    """Captures all artifacts produced by run_seed_sweep.

    Attributes
    ----------
    df : pandas.DataFrame
        One row per seed. Columns:
          seed,
          f1_macro, f1_weighted, f1_nonhab, f1_meso, f1_psychro,
          prec_macro, prec_weighted, prec_nonhab, prec_meso, prec_psychro,
          recall_macro, recall_weighted, recall_nonhab, recall_meso, recall_psychro,
          accuracy, balanced_acc, mcc, minority_f1_macro.
    cms : list of np.ndarray
        Count confusion matrices (3,3, int) per seed, ordered by classes
        [Non-Habitable, Mesoplanet, Psychroplanet].
    y_test_per_seed, y_pred_per_seed : list of np.ndarray
        Per-seed held-out labels and model predictions.
    feature_importances : list of np.ndarray, or None
        Per-seed feature_importances_ vectors, when the estimator exposes it.
        None for models without the attribute.
    feature_names : list of str
        Column names corresponding to importance vectors.
    model_name : str
        Label used in printouts and plot titles.
    """
    df: pd.DataFrame
    cms: List[np.ndarray] = field(default_factory=list)
    y_test_per_seed: List[np.ndarray] = field(default_factory=list)
    y_pred_per_seed: List[np.ndarray] = field(default_factory=list)
    feature_importances: Optional[List[np.ndarray]] = None
    feature_names: List[str] = field(default_factory=list)
    model_name: str = "model"

    @property
    def n_seeds(self) -> int:
        return len(self.df)


def _resolve_sweep_path(path):
    """Bare filenames land in data/output/seed_sweep_results/.
    Absolute paths or paths with parent components are used verbatim.
    """
    p = Path(path)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return SWEEP_DIR / p


def run_seed_sweep(model_factory, df, *, n_seeds=50, scaler_cls=None,
                   features=None, smote_variant="standard",
                   model_name="model", verbose=True,
                   save_csv=None, save_pickle=None):
    """Multi-seed train/test sweep with fixed hyperparameters.

    Fits a fresh model per seed via `model_factory(class_weight_dict)`, records
    F1/precision/recall (per-class, macro, weighted), accuracy, balanced
    accuracy, MCC, minority-F1-macro, plus the confusion matrix, predictions,
    and (when available) feature importances. Returns a SweepResult.

    Save paths: a bare filename for `save_csv` / `save_pickle` lands in
    `data/output/seed_sweep_results/`. Pass an absolute path or one with
    parent components to override.

    Parameters
    ----------
    model_factory : callable(class_weight_dict) -> sklearn estimator
        Returns a fresh estimator per seed. Models that don't accept
        class_weight (GNB, KNN, LDA, QDA, MLP, XGB) can ignore the argument.
    df : pandas.DataFrame
        Post-physics-imputation frame. Tier 3-7 imputers are re-fit on each
        seed's training split inside split_and_balance().
    n_seeds : int
        Number of random_state values to sweep (0 .. n_seeds-1). Default 50
        for fast iteration; bump to 200-500 for final results.
    scaler_cls : class, optional
        StandardScaler (default) or RobustScaler. Fresh instance per seed.
    features : list of str, optional
        Column subset (e.g. GNB drops the three derived columns).
    smote_variant : {"standard", "borderline"}
    save_csv, save_pickle : str or Path, optional
        Filenames or paths. Bare filenames land in SWEEP_DIR.

    Returns
    -------
    SweepResult
    """
    if scaler_cls is None:
        scaler_cls = StandardScaler

    _, classes, _ = get_class_meta()
    rows = []
    cms = []
    y_test_per_seed = []
    y_pred_per_seed = []
    importances_per_seed = []
    feature_names = None
    has_importances = None

    for seed in range(n_seeds):
        kwargs = {"scaler": scaler_cls(), "smote_variant": smote_variant,
                  "random_state": seed, "verbose": False}
        if features is not None:
            kwargs["features"] = features

        X_tr, y_tr, X_te, y_te, cw, _ = split_and_balance(df, **kwargs)
        if feature_names is None:
            feature_names = list(X_tr.columns)

        est = model_factory(cw)
        est.fit(X_tr, y_tr)
        y_pred = est.predict(X_te)

        # Per-class, macro, and weighted P/R/F1 in three calls
        p_per, r_per, f_per, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, average=None, zero_division=0,
        )
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, average="macro", zero_division=0,
        )
        p_wtd, r_wtd, f_wtd, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, average="weighted", zero_division=0,
        )

        rows.append({
            "seed":              seed,
            "f1_macro":          f_macro,
            "f1_weighted":       f_wtd,
            "f1_nonhab":         f_per[0],
            "f1_meso":           f_per[1],
            "f1_psychro":        f_per[2],
            "prec_macro":        p_macro,
            "prec_weighted":     p_wtd,
            "prec_nonhab":       p_per[0],
            "prec_meso":         p_per[1],
            "prec_psychro":      p_per[2],
            "recall_macro":      r_macro,
            "recall_weighted":   r_wtd,
            "recall_nonhab":     r_per[0],
            "recall_meso":       r_per[1],
            "recall_psychro":    r_per[2],
            "accuracy":          accuracy_score(y_te, y_pred),
            "balanced_acc":      balanced_accuracy_score(y_te, y_pred),
            "mcc":               matthews_corrcoef(y_te, y_pred),
            "minority_f1_macro": (f_per[1] + f_per[2]) / 2.0,
        })

        cms.append(confusion_matrix(y_te, y_pred, labels=classes))
        y_test_per_seed.append(np.asarray(y_te))
        y_pred_per_seed.append(np.asarray(y_pred))

        if has_importances is None:
            has_importances = hasattr(est, "feature_importances_")
        if has_importances:
            importances_per_seed.append(np.asarray(est.feature_importances_))

    sweep_df = pd.DataFrame(rows)
    result = SweepResult(
        df=sweep_df,
        cms=cms,
        y_test_per_seed=y_test_per_seed,
        y_pred_per_seed=y_pred_per_seed,
        feature_importances=importances_per_seed if has_importances else None,
        feature_names=feature_names or [],
        model_name=model_name,
    )

    if verbose:
        m, s = sweep_df["f1_macro"].mean(), sweep_df["f1_macro"].std()
        sem = s / np.sqrt(len(sweep_df))
        print(f"{model_name} — {n_seeds}-seed sweep")
        print(f"  F1 macro:    {m:.4f} ± {s:.4f}  (SEM {sem:.4f}, "
              f"min {sweep_df['f1_macro'].min():.4f}, "
              f"max {sweep_df['f1_macro'].max():.4f})")
        print(f"  F1 weighted: {sweep_df['f1_weighted'].mean():.4f} "
              f"± {sweep_df['f1_weighted'].std():.4f}")
        if has_importances:
            print(f"  Captured feature importances for {len(importances_per_seed)} seeds.")

    if save_csv is not None:
        path = _resolve_sweep_path(save_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(path, index=False)
        if verbose:
            print(f"  F1 DataFrame saved to {path}")

    if save_pickle is not None:
        import pickle
        path = _resolve_sweep_path(save_pickle)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        if verbose:
            print(f"  Full SweepResult pickled to {path}")

    return result


# ── Sweep-aware report helpers ──

def print_sweep_classification_report(result):
    """Per-class F1 mean ± std (min, max) across seeds — sklearn-style header."""
    class_names, _, _ = get_class_meta()
    suffix_for = {"Non-Habitable": "nonhab",
                  "Mesoplanet":    "meso",
                  "Psychroplanet": "psychro"}
    sweep_df = result.df
    n = result.n_seeds
    name = result.model_name

    print("=" * 64)
    print(f"{name} — Classification Report ({n}-seed sweep)")
    print("=" * 64)
    print(f"{'Class':<16} {'F1 mean':>10} {'F1 std':>10} {'F1 min':>10} {'F1 max':>10}")
    print("-" * 64)
    for cls in class_names:
        s = sweep_df[f"f1_{suffix_for[cls]}"]
        print(f"{cls:<16} {s.mean():>10.4f} {s.std():>10.4f} "
              f"{s.min():>10.4f} {s.max():>10.4f}")
    print("-" * 64)
    for label, col in [("Macro avg", "f1_macro"), ("Weighted avg", "f1_weighted")]:
        s = sweep_df[col]
        print(f"{label:<16} {s.mean():>10.4f} {s.std():>10.4f} "
              f"{s.min():>10.4f} {s.max():>10.4f}")


def print_sweep_performance_summary(result):
    """Cross-model performance metrics: per-class P/R + aggregate robustness.

    Designed to be called after run_stratified_cv in the Metrics cell. Reveals
    failure modes that F1 alone hides (precision vs. recall tradeoffs) and
    provides single-number summaries (MCC, minority F1 macro, balanced
    accuracy) suitable for cross-model ranking.
    """
    class_names, _, _ = get_class_meta()
    suffix_for = {"Non-Habitable": "nonhab",
                  "Mesoplanet":    "meso",
                  "Psychroplanet": "psychro"}
    sweep_df = result.df
    n = result.n_seeds
    name = result.model_name

    print()
    print("=" * 64)
    print(f"{name} — Performance Summary ({n}-seed sweep)")
    print("=" * 64)
    print()
    print("Per-class Precision and Recall (mean ± std)")
    print(f"{'':<16} {'Precision':<20} {'Recall':<20}")
    for cls in class_names:
        s = suffix_for[cls]
        p_m, p_s = sweep_df[f"prec_{s}"].mean(),   sweep_df[f"prec_{s}"].std()
        r_m, r_s = sweep_df[f"recall_{s}"].mean(), sweep_df[f"recall_{s}"].std()
        print(f"{cls:<16} {p_m:.4f} ± {p_s:.4f}    {r_m:.4f} ± {r_s:.4f}")
    print("-" * 64)
    for label, prefix in [("Macro avg", "macro"), ("Weighted avg", "weighted")]:
        p_m, p_s = sweep_df[f"prec_{prefix}"].mean(),   sweep_df[f"prec_{prefix}"].std()
        r_m, r_s = sweep_df[f"recall_{prefix}"].mean(), sweep_df[f"recall_{prefix}"].std()
        print(f"{label:<16} {p_m:.4f} ± {p_s:.4f}    {r_m:.4f} ± {r_s:.4f}")

    print()
    print("Aggregate Robustness Metrics")
    print("-" * 64)
    rows = [
        ("Accuracy",             "accuracy",          ""),
        ("Balanced accuracy",    "balanced_acc",      "(= macro recall)"),
        ("Matthews Corr. Coef.", "mcc",               "(−1 to +1; 0 = chance)"),
        ("Minority F1 macro",    "minority_f1_macro", "(meso + psychro only)"),
    ]
    for label, col, note in rows:
        m, s = sweep_df[col].mean(), sweep_df[col].std()
        print(f"  {label:<22} {m:.4f} ± {s:.4f}    {note}")


def print_sweep_error_patterns(result):
    """Off-diagonal confusion cells aggregated across seeds.

    For each (true, predicted) pair with non-zero occurrence, prints both the
    per-split count and the per-class rate as mean ± std.
    """
    class_names, _, _ = get_class_meta()
    cms = np.stack(result.cms)  # (n_seeds, 3, 3)
    n = result.n_seeds

    print(f"Key error patterns ({n}-seed sweep):")
    for i, true_name in enumerate(class_names):
        row_totals = cms[:, i, :].sum(axis=1).astype(float)
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            cell = cms[:, i, j]
            if cell.sum() == 0:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                rates = np.where(row_totals > 0, cell / row_totals, 0.0)
            print(
                f"  {true_name} → predicted {pred_name}: "
                f"{cell.mean():.2f} ± {cell.std():.2f} per split  "
                f"({rates.mean():.1%} ± {rates.std():.1%} of true {true_name})"
            )
