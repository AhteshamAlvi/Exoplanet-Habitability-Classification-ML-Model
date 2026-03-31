# dependencies.py — Central import file for all project modules and notebooks

# ── Core ──
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

# ── Visualization ──
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Stats ──
from scipy.stats import norm

# ── Sklearn: Preprocessing ──
from sklearn.experimental import enable_iterative_imputer  # noqa: must come before IterativeImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import IterativeImputer, KNNImputer

# ── Sklearn: Model Selection ──
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV,
)

# ── Sklearn: Metrics ──
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, ConfusionMatrixDisplay,
)

# ── Sklearn: Models ──
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# ── XGBoost ──
from xgboost import XGBClassifier

# ── Imbalanced-learn ──
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline

PROJECT_ROOT = Path(__file__).parent
FIG_DIR = PROJECT_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)  # creates folder if missing

def save_plot(name):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")