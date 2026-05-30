"""
plots.py — Visualisation helpers for the exoplanet ML pipeline.

The sweep-aware plots are the main output of the new pipeline:
  - plot_sweep_f1_bar              : per-class F1 bar chart with error bars
  - plot_sweep_confusion_matrix    : averaged normalized CM with per-cell std
  - plot_sweep_feature_importances : mean ± std importance per feature

Single-split plots retained for the tuned seed=42 model:
  - plot_posterior_violins         : predicted-probability distributions
                                     (no meaningful sweep aggregation)
"""
import numpy as np
import matplotlib.pyplot as plt

from .constants import get_class_meta
from .dependencies import save_plot


def plot_posterior_violins(y_test, y_prob, model_name):
    """Violin + jitter plot of predicted class probabilities, by true class.

    Only applicable to models that expose predict_proba (GNB, LogReg, QDA, MLP).
    When a true class has ≤50 samples the individual points are also
    scatter-plotted over the violin. Reports for a single split (seed=42)
    because per-instance probability distributions don't average meaningfully.
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

    # Per-seed row-normalized CMs, then aggregate
    norms = []
    for cm in result.cms:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        norms.append(np.nan_to_num(cm_norm))
    arr = np.stack(norms)
    mean_norm = arr.mean(axis=0)
    std_norm = arr.std(axis=0)

    # Display order
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
    (linear models, SVMs, KNN, Naïve Bayes). Pass an optional `subtitle` to
    annotate the plot with hyperparameter context, e.g. tree depth.
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
