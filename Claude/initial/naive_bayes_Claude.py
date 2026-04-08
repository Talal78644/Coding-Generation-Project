"""
Gaussian Naive Bayes — AI4I 2020 Predictive Maintenance Dataset
================================================================
Pipeline:
  1. Load & clean data
  2. Feature engineering (Type encoding, temp delta)
  3. Train/test split  →  SMOTE oversampling on train only
  4. StandardScaler normalisation
  5. GaussianNB training
  6. Outputs:
       Fig 1 – Performance metrics bar chart (Precision / Recall / F1 / ROC-AUC)
       Fig 2 – ROC Curve
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────
# 0.  Style
# ──────────────────────────────────────────────
PALETTE = {
    "precision": "#4C72B0",
    "recall":    "#DD8452",
    "f1":        "#55A868",
    "roc_auc":   "#C44E52",
    "bg":        "#F8F9FA",
    "grid":      "#DEE2E6",
    "text":      "#212529",
    "accent":    "#6C757D",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

# ──────────────────────────────────────────────
# 1.  Load data
# ──────────────────────────────────────────────
df = pd.read_csv("ai4i2020.csv")

# Drop identifier / sub-failure columns
df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        inplace=True)

# ──────────────────────────────────────────────
# 2.  Feature engineering
# ──────────────────────────────────────────────
# Ordinal-encode product quality type  (L=0, M=1, H=2)
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Derived feature: temperature difference
df["Temp_delta [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

# Power feature
df["Power [W]"] = (df["Torque [Nm]"] * df["Rotational speed [rpm]"]
                   * 2 * np.pi / 60)

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Temp_delta [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power [W]",
]
TARGET = "Machine failure"

X = df[FEATURES].values
y = df[TARGET].values

print(f"Dataset shape : {X.shape}")
print(f"Class balance : {np.bincount(y)}  →  failure rate {y.mean()*100:.2f}%")

# ──────────────────────────────────────────────
# 3.  Train / test split
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 4.  SMOTE  (applied to train set only)
# ──────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE   : {np.bincount(y_train_res)}")

# ──────────────────────────────────────────────
# 5.  Scaling
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_res)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 6.  Gaussian Naive Bayes
# ──────────────────────────────────────────────
gnb = GaussianNB()
gnb.fit(X_train_sc, y_train_res)

y_pred      = gnb.predict(X_test_sc)
y_prob      = gnb.predict_proba(X_test_sc)[:, 1]

# ──────────────────────────────────────────────
# 7.  Metrics
# ──────────────────────────────────────────────
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_prob)

print("\n── Classification Report ─────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
print(f"ROC-AUC : {roc_auc:.4f}")

# Cross-val ROC-AUC on the full resampled set (5-fold)
cv_scores = cross_val_score(
    GaussianNB(), X_train_sc, y_train_res,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
)
print(f"5-Fold CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ROC curve data
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_curve = auc(fpr, tpr)

# ──────────────────────────────────────────────
# 8.  Figure 1 – Performance Metrics Bar Chart
# ──────────────────────────────────────────────
metrics      = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
values       = [precision, recall, f1, roc_auc]
bar_colours  = [PALETTE["precision"], PALETTE["recall"],
                PALETTE["f1"], PALETTE["roc_auc"]]

fig1, ax1 = plt.subplots(figsize=(9, 5.5))
fig1.patch.set_facecolor(PALETTE["bg"])

bars = ax1.bar(metrics, values, color=bar_colours,
               width=0.5, edgecolor="white", linewidth=1.2,
               zorder=3)

# Value labels on bars
for bar, val in zip(bars, values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.012,
        f"{val:.3f}",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color=PALETTE["text"],
    )

ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Score", fontsize=13, labelpad=8)
ax1.set_title(
    "Gaussian Naive Bayes — Performance Metrics\nAI4I 2020 Predictive Maintenance",
    fontsize=14, fontweight="bold", pad=14,
)
ax1.axhline(0.5, color=PALETTE["accent"], linewidth=0.9,
            linestyle=":", alpha=0.8, label="0.5 baseline", zorder=2)
ax1.legend(fontsize=10)
ax1.grid(axis="y", zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", labelsize=12)
ax1.spines[["top", "right"]].set_visible(False)

# Colour-coded x-tick labels
for tick, colour in zip(ax1.get_xticklabels(), bar_colours):
    tick.set_color(colour)
    tick.set_fontweight("bold")

plt.tight_layout()
fig1.savefig("nb_performance_metrics.png",
             dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
print("Saved: nb_performance_metrics.png")

# ──────────────────────────────────────────────
# 9.  Figure 2 – ROC Curve
# ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor(PALETTE["bg"])

ax2.plot(fpr, tpr, color=PALETTE["roc_auc"], lw=2.5,
         label=f"Gaussian NB  (AUC = {roc_auc_curve:.3f})")
ax2.plot([0, 1], [0, 1], color=PALETTE["accent"],
         lw=1.4, linestyle="--", label="Random classifier")

ax2.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["roc_auc"])

ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate", fontsize=13)
ax2.set_ylabel("True Positive Rate", fontsize=13)
ax2.set_title(
    "ROC Curve — Gaussian Naive Bayes\nAI4I 2020 Predictive Maintenance",
    fontsize=14, fontweight="bold", pad=14,
)
ax2.legend(loc="lower right", fontsize=11)
ax2.grid(True, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig2.savefig("nb_roc_curve.png",
             dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
print("Saved: nb_roc_curve.png")

plt.show()
print("\n✓ All figures saved to /mnt/user-data/outputs/")
