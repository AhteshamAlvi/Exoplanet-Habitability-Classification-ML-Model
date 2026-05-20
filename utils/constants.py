import numpy as np


CLASS_NAMES = ["Non-Habitable", "Mesoplanet", "Psychroplanet"]
CLASSES = np.array([0.0, 1.0, 2.0])
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


def get_class_meta():
    """Return (class_names, classes, colors) shared across all models."""
    return CLASS_NAMES, CLASSES, COLORS
