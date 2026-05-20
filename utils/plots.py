import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .constants import get_class_meta
from .evaluation import compute_f1_scores

_FIG_DIR = Path(__file__).parent.parent / "report" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(name):
    plt.savefig(_FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_f1_bar(y_test, y_pred, model_name):
    """Bar chart of per-class F1 with macro and weighted reference lines."""
    class_names, _, colors = get_class_meta()
    f1_macro, f1_weighted, f1_per_class = compute_f1_scores(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(class_names))
    bars = ax.bar(x_pos, f1_per_class, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, score in zip(bars, f1_per_class):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{score:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.axhline(f1_macro, color="gray", ls="--", lw=1.5, label=f"Macro Avg = {f1_macro:.3f}")
    ax.axhline(f1_weighted, color="gray", ls=":", lw=1.5, label=f"Weighted Avg = {f1_weighted:.3f}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name} — Per-Class F1 Scores", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(f"{model_name.lower().replace(' ', '_')}_f1_scores")
    plt.show()


def plot_confusion_matrices(y_test, y_pred, model_name):
    """Side-by-side raw counts and row-normalised confusion matrices.

    Rows are ordered Psychroplanet → Mesoplanet → Non-Habitable (rarest first)
    so that habitable classes sit at the top for easier reading.
    """
    class_names, classes, _ = get_class_meta()
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    row_order = [2, 1, 0]
    col_order = [0, 1, 2]
    cm_r = cm[np.ix_(row_order, col_order)]
    cm_norm_r = cm_norm[np.ix_(row_order, col_order)]
    y_labels = [class_names[i] for i in row_order]
    x_labels = [class_names[i] for i in col_order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    disp1 = ConfusionMatrixDisplay(cm_r, display_labels=x_labels)
    disp1.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_yticklabels(y_labels)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=12, fontweight="bold")

    disp2 = ConfusionMatrixDisplay(cm_norm_r, display_labels=x_labels)
    disp2.plot(ax=axes[1], cmap="Oranges", colorbar=False, values_format=".2%")
    axes[1].set_yticklabels(y_labels)
    axes[1].set_title("Confusion Matrix (Row-Normalized)", fontsize=12, fontweight="bold")

    fig.suptitle(f"{model_name} — Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(f"{model_name.lower().replace(' ', '_')}_confusion_matrix")
    plt.show()


def plot_posterior_violins(y_test, y_prob, model_name):
    """Violin + jitter plot of predicted class probabilities broken out by true class.

    Only applicable to models that expose predict_proba. When a true class has ≤50
    samples the individual points are also scatter-plotted over the violin.
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
    _save(f"{model_name.lower().replace(' ', '_')}_posteriors")
    plt.show()
