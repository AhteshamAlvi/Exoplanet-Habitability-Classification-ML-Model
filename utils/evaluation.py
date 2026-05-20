import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .constants import get_class_meta


def print_classification_report(y_test, y_pred, model_name):
    """Print a labelled classification report for the three habitability classes."""
    class_names, _, _ = get_class_meta()
    print("=" * 60)
    print(f"{model_name} — Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))


def compute_f1_scores(y_test, y_pred):
    """Return (f1_macro, f1_weighted, f1_per_class) for the test predictions."""
    _, classes, _ = get_class_meta()
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=classes, zero_division=0)
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    return f1_macro, f1_weighted, f1_per_class


def run_stratified_cv(model, X_train, y_train, n_splits=5):
    """Run stratified k-fold CV scored on f1_macro. Prints mean ± std and returns scores array."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"\n{n_splits}-Fold CV F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def print_error_patterns(y_test, y_pred):
    """Print every off-diagonal confusion cell as 'TrueClass → PredClass: N (X%)'."""
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
