"""
plots.py — Visualisation helpers for the exoplanet ML pipeline.

Sweep-aware plots — main outputs of the new pipeline:
  - plot_sweep_f1_bar              : per-class F1 bar chart with error bars
  - plot_sweep_confusion_matrix    : averaged normalized CM with per-cell std
  - plot_sweep_feature_importances : mean ± std importance per feature
  - plot_sweep_posterior_violins   : aggregated predict_proba distributions
                                     (across all seeds, far more statistical
                                     mass than a single split)
  - plot_sweep_loss_curves         : overlaid per-seed loss curves with a
                                     bold mean — reveals init-sensitivity

Single-split plot retained:
  - plot_posterior_violins         : seed=42 predict_proba violins (kept for
                                     models that prefer a per-split view)
"""
import numpy as np
import matplotlib.pyplot as plt

from .constants import get_class_meta
from .dependencies import save_plot


def plot_posterior_violins(y_test, y_prob, model_name):
    """Violin + jitter plot of predicted class probabilities, by true class.

    Single-seed=42 view, only applicable to models that expose predict_proba
    (GNB, LogReg, QDA, MLP). When a true class has ≤50 samples the individual
    points are also scatter-plotted over the violin. For the sweep-aware
    aggregated version, use plot_sweep_posterior_violins.
    """
    class_names, _, colors = get_class_meta()
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for true_cls in range(3):
        ax = axes[true_cls]
        mask = y_test_arr == float(true_cls)
        probs = y_prob[mask]

        if probs.shape[0] == 0:
            ax.set_title(f"True: {class_names[true_cls]}\n(no samples in test set)")
            continue

        parts = ax.violinplot(
            [probs[:, c] for c in range(3)],
            positions=[0, 1, 2],
            showmeans=True,
            showmedians=True,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        for key in ("cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color("black")

        if probs.shape[0] <= 50:
            rng = np.random.default_rng(42)
            for c in range(3):
                jitter = rng.normal(0, 0.04, size=probs.shape[0])
                ax.scatter(
                    np.full(probs.shape[0], c) + jitter,
                    probs[:, c],
                    color=colors[c], s=25, alpha=0.7,
                    edgecolors="black", linewidth=0.5, zorder=5,
                )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["P(Non-Hab)", "P(Meso)", "P(Psychro)"], fontsize=9)
        ax.set_ylabel("Predicted Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"True Class: {class_names[true_cls]}\n(n = {mask.sum()})",
            fontsize=11, fontweight="bold",
        )
        ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)

    fig.suptitle(
        f"{model_name} — Posterior Probabilities by True Class",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_plot(f"{model_name.lower().replace(' ', '_')}_posteriors")
    plt.show()


# ── Sweep-aware visualisations ──

def plot_sweep_f1_bar(result):
    """Per-class F1 bar chart with error bars (mean ± std across seeds)."""
    class_names, _, colors = get_class_meta()
    col_for = {"Non-Habitable": "f1_nonhab",
               "Mesoplanet":    "f1_meso",
               "Psychroplanet": "f1_psychro"}
    sweep_df = result.df
    means = [sweep_df[col_for[c]].mean() for c in class_names]
    stds  = [sweep_df[col_for[c]].std()  for c in class_names]
    macro_mean = sweep_df["f1_macro"].mean()
    macro_std  = sweep_df["f1_macro"].std()
    weighted_mean = sweep_df["f1_weighted"].mean()
    n = result.n_seeds
    name = result.model_name

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5, capsize=6, error_kw={"lw": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.axhline(macro_mean, color="gray", ls="--", lw=1.5,
               label=f"Macro avg = {macro_mean:.3f} ± {macro_std:.3f}")
    ax.axhline(weighted_mean, color="gray", ls=":", lw=1.5,
               label=f"Weighted avg = {weighted_mean:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{name} — Per-Class F1 ({n}-seed sweep, mean ± std)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_f1_scores_sweep")
    plt.show()


def plot_sweep_confusion_matrix(result):
    """Average row-normalized confusion matrix across seeds, annotated with std.

    Rows are ordered Psychroplanet → Mesoplanet → Non-Habitable (rarest first)
    so the habitable classes sit at the top, matching the single-split layout.
    The count CM is intentionally omitted — averaged integer counts are
    fractional and less interpretable than the percentage view.
    """
    class_names, _, _ = get_class_meta()
    n = result.n_seeds
    name = result.model_name

    norms = []
    for cm in result.cms:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        norms.append(np.nan_to_num(cm_norm))
    arr = np.stack(norms)
    mean_norm = arr.mean(axis=0)
    std_norm = arr.std(axis=0)

    row_order = [2, 1, 0]
    col_order = [0, 1, 2]
    mean_r = mean_norm[np.ix_(row_order, col_order)]
    std_r = std_norm[np.ix_(row_order, col_order)]
    y_labels = [class_names[i] for i in row_order]
    x_labels = [class_names[i] for i in col_order]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mean_r, cmap="Oranges", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if mean_r[i, j] > 0.5 else "black"
            ax.text(j, i, f"{mean_r[i,j]:.1%}\n±{std_r[i,j]:.1%}",
                    ha="center", va="center", color=color, fontsize=11)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(
        f"{name} — Average Normalized Confusion Matrix\n"
        f"({n}-seed sweep, mean ± std per cell)",
        fontsize=13, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Row-normalized rate", fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_confusion_matrix_sweep")
    plt.show()


def plot_sweep_feature_importances(result, subtitle=None):
    """Per-feature importance bars (mean ± std across seeds).

    Silently returns if the swept model didn't expose feature_importances_
    (linear models, SVMs, KNN, Naïve Bayes, MLP). Pass an optional `subtitle`
    to annotate the plot with hyperparameter context.
    """
    if result.feature_importances is None:
        print(f"[{result.model_name}] No feature_importances_ available "
              f"for this estimator — skipping importance plot.")
        return

    arr = np.stack(result.feature_importances)  # (n_seeds, n_features)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    n = result.n_seeds
    names = result.feature_names
    name = result.model_name

    sorted_idx = np.argsort(means)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(names))
    ax.barh(
        y_pos, means[sorted_idx], xerr=stds[sorted_idx],
        color="#2196F3", ecolor="gray", capsize=3, edgecolor="none",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in sorted_idx], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance (mean ± std across seeds)", fontsize=11)
    title = f"{name} — Feature Importances ({n}-seed sweep)"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_feature_importances_sweep")
    plt.show()


def plot_sweep_posterior_violins(result):
    """Aggregated posterior probability violins across all sweep seeds.

    Concatenates predict_proba across all seeds, gaining substantial statistical
    mass for minority classes (e.g., 50 splits × 6 mesoplanet test samples =
    300 probability points per panel). Silently returns if predict_proba wasn't
    captured (SVMs without probability=True, KNN, etc.).
    """
    if result.y_prob_per_seed is None or not result.y_prob_per_seed:
        print(f"[{result.model_name}] No predict_proba captured — skipping "
              f"sweep-aware posterior violins.")
        return

    class_names, _, colors = get_class_meta()
    name = result.model_name
    n = result.n_seeds

    y_test_all = np.concatenate([np.asarray(y) for y in result.y_test_per_seed])
    y_prob_all = np.concatenate(result.y_prob_per_seed, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for true_cls in range(3):
        ax = axes[true_cls]
        mask = y_test_all == float(true_cls)
        probs = y_prob_all[mask]
        n_samples = int(mask.sum())

        if n_samples == 0:
            ax.set_title(f"True: {class_names[true_cls]}\n(no samples)")
            continue

        parts = ax.violinplot(
            [probs[:, c] for c in range(3)],
            positions=[0, 1, 2],
            showmeans=True, showmedians=True,
        )
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        for key in ("cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color("black")

        # Scatter the individual points when they're still readable (≤500)
        if n_samples <= 500:
            rng = np.random.default_rng(42)
            for c in range(3):
                jitter = rng.normal(0, 0.04, size=n_samples)
                ax.scatter(
                    np.full(n_samples, c) + jitter,
                    probs[:, c],
                    color=colors[c], s=8, alpha=0.35,
                    edgecolors="black", linewidth=0.2, zorder=5,
                )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["P(Non-Hab)", "P(Meso)", "P(Psychro)"], fontsize=9)
        ax.set_ylabel("Predicted Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"True Class: {class_names[true_cls]}\n"
            f"(n = {n_samples} across {n} seeds)",
            fontsize=11, fontweight="bold",
        )
        ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)

    fig.suptitle(
        f"{name} — Posterior Probabilities by True Class ({n}-seed sweep)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_posteriors_sweep")
    plt.show()


def plot_sweep_linear_coefficients(result, panel_labels=None):
    """Per-OvR-boundary coefficient bars with mean ± std across seeds.

    For models with coef_ of shape (n_classifiers, n_features) (linear SVM,
    LogReg). Each panel shows one classifier's feature weights sorted by mean
    magnitude, color-coded green/red by sign of the mean. Error bars are the
    std of each coefficient across seeds.

    OvR multiclass: signs are semantically consistent across seeds (e.g.,
    "Non-Hab vs Rest" is always the same binary problem) so signed averaging
    is meaningful — unlike LDA's discriminant axes which suffer sign ambiguity.

    Silently returns if coef_ wasn't captured.

    Parameters
    ----------
    result : SweepResult
    panel_labels : list of str, optional
        One label per classifier. Defaults to "<ClassName> vs Rest" when the
        first dimension of coef_ matches the number of classes.
    """
    if result.coefs is None or not result.coefs:
        print(f"[{result.model_name}] No coef_ captured — skipping "
              f"sweep-aware linear coefficient plot.")
        return

    class_names, _, _ = get_class_meta()
    feature_names = result.feature_names
    name = result.model_name
    n = result.n_seeds

    arr = np.stack(result.coefs)  # could be (n_seeds, n_clf, n_features) or (n_seeds, n_features)
    if arr.ndim == 2:
        arr = arr[:, np.newaxis, :]  # promote binary case to (n_seeds, 1, n_features)
    n_classifiers = arr.shape[1]
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)

    if panel_labels is None:
        if n_classifiers == len(class_names):
            panel_labels = [f"{c} vs Rest" for c in class_names]
        else:
            panel_labels = [f"Classifier {i+1}" for i in range(n_classifiers)]

    fig, axes = plt.subplots(1, n_classifiers, figsize=(6 * n_classifiers, 7))
    if n_classifiers == 1:
        axes = [axes]

    for cls_idx, ax in enumerate(axes):
        m = means[cls_idx]
        s = stds[cls_idx]
        sorted_idx = np.argsort(np.abs(m))[::-1]
        sorted_m = m[sorted_idx]
        sorted_s = s[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        bar_colors = ["#4CAF50" if w > 0 else "#F44336" for w in sorted_m]

        ax.barh(range(len(sorted_names)), sorted_m, xerr=sorted_s,
                color=bar_colors, ecolor="gray", capsize=3, edgecolor="none")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Weight (mean ± std across seeds)", fontsize=11)
        ax.set_title(panel_labels[cls_idx], fontsize=12, fontweight="bold")

    fig.suptitle(
        f"{name} — OvR Feature Coefficients ({n}-seed sweep, mean ± std)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_coefficients_sweep")
    plt.show()


def plot_sweep_decision_scores(result):
    """Aggregated decision-function score violins by true class across all seeds.

    SVM/linear-classifier analog of plot_sweep_posterior_violins. For each true
    class, shows the distribution of decision scores for each OvR classifier,
    aggregated across all sweep seeds — yielding 50 × 6 = 300 mesoplanet score
    points per minority panel, real statistical mass. Uses signed margin
    distance (decision_function), SVM's native output, rather than the Platt-
    scaled predict_proba which is unreliable on small minority classes.

    Silently returns if decision_function wasn't captured.
    """
    if result.y_dec_per_seed is None or not result.y_dec_per_seed:
        print(f"[{result.model_name}] No decision_function captured — skipping "
              f"sweep-aware decision score plot.")
        return

    class_names, _, colors = get_class_meta()
    name = result.model_name
    n = result.n_seeds

    y_test_all = np.concatenate([np.asarray(y) for y in result.y_test_per_seed])
    y_dec_all = np.concatenate(result.y_dec_per_seed, axis=0)

    # Binary case: decision_function returns shape (n_samples,) instead of (n_samples, 1)
    if y_dec_all.ndim == 1:
        y_dec_all = y_dec_all[:, np.newaxis]
    n_panels = y_dec_all.shape[1]
    score_labels = [f"Score: {c[:8]}" for c in class_names[:n_panels]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    y_min = float(y_dec_all.min()) - 0.1
    y_max = float(y_dec_all.max()) + 0.1

    for true_cls in range(3):
        ax = axes[true_cls]
        mask = y_test_all == float(true_cls)
        scores = y_dec_all[mask]
        n_samples = int(mask.sum())

        if n_samples == 0:
            ax.set_title(f"True: {class_names[true_cls]}\n(no samples)")
            continue

        parts = ax.violinplot(
            [scores[:, c] for c in range(n_panels)],
            positions=list(range(n_panels)),
            showmeans=True, showmedians=True,
        )
        for pc, color in zip(parts["bodies"], colors[:n_panels]):
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
        for key in ("cmeans", "cmedians"):
            if key in parts:
                parts[key].set_color("black")

        if n_samples <= 500:
            rng = np.random.default_rng(42)
            for c in range(n_panels):
                jitter = rng.normal(0, 0.04, size=n_samples)
                ax.scatter(
                    np.full(n_samples, c) + jitter,
                    scores[:, c],
                    color=colors[c], s=8, alpha=0.35,
                    edgecolors="black", linewidth=0.2, zorder=5,
                )

        ax.set_xticks(list(range(n_panels)))
        ax.set_xticklabels(score_labels, fontsize=9)
        ax.set_ylabel("Decision Score (signed margin distance)")
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_title(
            f"True Class: {class_names[true_cls]}\n"
            f"(n = {n_samples} across {n} seeds)",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"{name} — Decision Function Scores by True Class ({n}-seed sweep)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_decision_scores_sweep")
    plt.show()


def plot_sweep_loss_curves(result):
    """Overlay every per-seed training loss curve, with a bold mean curve on top.

    Reveals init-sensitivity for iterative learners (MLP especially): a tight
    bundle = consistent convergence, a wide spread = the optimizer found
    different basins from different seeds. Title reports the distribution of
    n_iter (median, min, max). Silently returns if no loss curves were captured.
    """
    if result.loss_curves is None or not result.loss_curves:
        print(f"[{result.model_name}] No loss_curve_ captured — skipping "
              f"sweep-aware loss curve plot.")
        return

    curves = result.loss_curves
    name = result.model_name
    n = result.n_seeds
    n_iters = result.n_iters or [len(c) for c in curves]

    # Pad curves to the longest length with NaN so nanmean ignores absent epochs
    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    mean_curve = np.nanmean(padded, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for c in curves:
        ax.plot(c, color="#2196F3", alpha=0.15, lw=1)
    ax.plot(mean_curve, color="#0D47A1", lw=2.5, label="Mean across seeds")
    ax.set_xlabel("Iteration (epoch)", fontsize=11)
    ax.set_ylabel("Loss (cross-entropy)", fontsize=11)
    ax.set_title(
        f"{name} — Training Loss Curves ({n}-seed sweep)\n"
        f"n_iter: median={int(np.median(n_iters))}, "
        f"min={int(min(n_iters))}, max={int(max(n_iters))}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_plot(f"{name.lower().replace(' ', '_')}_loss_curves_sweep")
    plt.show()
