import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────
# STYLE CONFIG
# ─────────────────────────────────────────────────────────────────────
PALETTE     = ['#2563EB', '#DC2626', '#16A34A']   # LR=blue, SVM=red, RF=green
BG          = '#0F172A'
PANEL       = '#1E293B'
TEXT        = '#F1F5F9'
GRID        = '#334155'
ACCENT      = '#38BDF8'
FAIL_CLR    = '#EF4444'
NO_FAIL_CLR = '#22D3EE'
OUTDIR      = ''

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.6)

def dark_fig(w=7, h=5):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def save(fig, name):
    path = OUTDIR + name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Saved → {name}')

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD & ENGINEER FEATURES
# ─────────────────────────────────────────────────────────────────────
print('\n══ Loading dataset ══')
df = pd.read_csv('ai4i2020.csv')

# Strip BOM / whitespace from column names
df.columns = df.columns.str.strip().str.lstrip('\ufeff')

print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Failure rate: {df["Machine failure"].mean():.2%}\n')

# Derived features
df['MechanicalPower']     = df['Torque [Nm]'] * 2 * np.pi * df['Rotational speed [rpm]'] / 60
df['WearTorqueInteract']  = df['Tool wear [min]'] * df['Torque [Nm]']
df['TempDifferential']    = df['Process temperature [K]'] - df['Air temperature [K]']

# One-hot encode Type
type_dummies = pd.get_dummies(df['Type'], prefix='Type', drop_first=False)

raw_features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]
derived_features = ['MechanicalPower', 'WearTorqueInteract', 'TempDifferential']

feature_names = raw_features + derived_features + list(type_dummies.columns)
X = pd.concat([df[raw_features + derived_features], type_dummies], axis=1)
y = df['Machine failure']

print(f'Features ({len(feature_names)}): {feature_names}')

# ─────────────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  →  SMOTE  →  SCALE
# ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f'\nAfter SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}')

# Scale numeric features only (not dummies)
numeric_cols  = raw_features + derived_features
dummy_cols    = list(type_dummies.columns)
num_idx       = [feature_names.index(c) for c in numeric_cols]

scaler = StandardScaler()
X_train_arr = np.array(X_train_sm, dtype=float)
X_test_arr  = np.array(X_test,    dtype=float)
X_train_arr[:, num_idx] = scaler.fit_transform(X_train_arr[:, num_idx])
X_test_arr[:,  num_idx] = scaler.transform(    X_test_arr[:,  num_idx])

# ─────────────────────────────────────────────────────────────────────
# HELPER: shared plots (ROC, PR/F1 bar)
# ─────────────────────────────────────────────────────────────────────
def plot_roc(y_true, y_score, label, color, title, fname):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)
    fig = dark_fig(6, 5)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f'{label}  (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1], color=GRID, lw=1.2, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.12, color=color)
    style_ax(ax, title=title, xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    save(fig, fname)
    return roc_auc, fpr, tpr

def plot_prf_bar(prec, rec, f1, title, color, fname):
    metrics = ['Precision', 'Recall', 'F1-Score']
    values  = [prec, rec, f1]
    fig = dark_fig(6, 4.5)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    bars = ax.bar(metrics, values, color=color, width=0.45,
                  edgecolor='white', linewidth=0.8, alpha=0.92)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012,
                f'{v:.3f}', ha='center', va='bottom',
                color=TEXT, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    style_ax(ax, title=title,
             xlabel='Metric', ylabel='Score (Failure Class)')
    fig.tight_layout()
    save(fig, fname)

# ─────────────────────────────────────────────────────────────────────
# 3. LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────
print('\n══ Logistic Regression ══')
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_arr, y_train_sm)

y_pred_lr   = lr.predict(X_test_arr)
y_prob_lr   = lr.predict_proba(X_test_arr)[:, 1]

prec_lr = precision_score(y_test, y_pred_lr)
rec_lr  = recall_score(y_test, y_pred_lr)
f1_lr   = f1_score(y_test, y_pred_lr)
acc_lr  = accuracy_score(y_test, y_pred_lr)

roc_lr, fpr_lr, tpr_lr = plot_roc(
    y_test, y_prob_lr, 'Logistic Regression', PALETTE[0],
    'Logistic Regression — ROC Curve',
    'logistic_regression_roc_curve.png')

plot_prf_bar(prec_lr, rec_lr, f1_lr,
    'Logistic Regression — Failure Class Metrics', PALETTE[0],
    'logistic_regression_prf_bar.png')

# (4) Sigmoid curve
print('  Plotting sigmoid…')
x_sig  = np.linspace(-6, 6, 400)
y_sig  = 1 / (1 + np.exp(-x_sig))

# Sample real linear-combination values for the overlay points
lc_vals = X_test_arr @ lr.coef_[0] + lr.intercept_[0]
lc_clip = np.clip(lc_vals, -6, 6)
p_overlay = 1 / (1 + np.exp(-lc_clip))

fig = dark_fig(7, 5)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL)
ax.plot(x_sig, y_sig, color=PALETTE[0], lw=2.8, label='σ(x)', zorder=3)
ax.scatter(lc_clip, p_overlay,
           c=PALETTE[1], s=18, alpha=0.55, zorder=4, label='Test-set points')
ax.axhline(0.5, color=GRID, linewidth=1.0, linestyle='--', alpha=0.7)
ax.axvline(0.0, color=GRID, linewidth=1.0, linestyle='--', alpha=0.7)
style_ax(ax,
    title='Logistic Regression — Sigmoid Function  σ(x) = 1/(1+e⁻ˣ)',
    xlabel='X',
    ylabel='sigmoid(X)')
ax.set_xlim(-6, 6); ax.set_ylim(-0.05, 1.05)
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
fig.tight_layout()
save(fig, 'logistic_regression_sigmoid.png')

# ─────────────────────────────────────────────────────────────────────
# 4. SVM (RBF kernel)
# ─────────────────────────────────────────────────────────────────────
print('\n══ SVM (RBF Kernel) ══')
svm = SVC(kernel='rbf', C=1.0, gamma='scale',
          probability=True, random_state=42)
svm.fit(X_train_arr, y_train_sm)

y_pred_svm  = svm.predict(X_test_arr)
y_prob_svm  = svm.predict_proba(X_test_arr)[:, 1]

prec_svm = precision_score(y_test, y_pred_svm)
rec_svm  = recall_score(y_test, y_pred_svm)
f1_svm   = f1_score(y_test, y_pred_svm)
acc_svm  = accuracy_score(y_test, y_pred_svm)

roc_svm, fpr_svm, tpr_svm = plot_roc(
    y_test, y_prob_svm, 'SVM (RBF)', PALETTE[1],
    'SVM (RBF) — ROC Curve',
    'svm_roc_curve.png')

plot_prf_bar(prec_svm, rec_svm, f1_svm,
    'SVM (RBF) — Failure Class Metrics', PALETTE[1],
    'svm_prf_bar.png')

# (4) 2-D decision boundary: Tool Wear vs Torque
print('  Plotting RBF decision boundary…')
feat_A_idx = feature_names.index('Tool wear [min]')
feat_B_idx = feature_names.index('Torque [Nm]')

# Build lightweight SVM on these 2 features only
X_tr2 = X_train_arr[:, [feat_A_idx, feat_B_idx]]
X_te2 = X_test_arr[:,  [feat_A_idx, feat_B_idx]]

svm2 = SVC(kernel='rbf', C=1.0, gamma='scale',
           probability=False, random_state=42)
svm2.fit(X_tr2, y_train_sm)

h = 0.04
x_min, x_max = X_te2[:, 0].min() - 0.5, X_te2[:, 0].max() + 0.5
y_min, y_max = X_te2[:, 1].min() - 0.5, X_te2[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = dark_fig(8, 6)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL)
ax.contourf(xx, yy, Z, alpha=0.30,
            cmap=plt.cm.RdBu_r, levels=[-0.5, 0.5, 1.5])
ax.contour(xx, yy, Z, levels=[0.5], colors='white',
           linewidths=1.5, linestyles='--', alpha=0.7)

colors_map = {0: NO_FAIL_CLR, 1: FAIL_CLR}
for cls, label in zip([0, 1], ['No Failure', 'Failure']):
    mask = (y_test.values == cls)
    ax.scatter(X_te2[mask, 0], X_te2[mask, 1],
               c=colors_map[cls], s=20, alpha=0.7,
               edgecolors='none', label=label, zorder=3)

style_ax(ax,
    title='SVM (RBF) — Decision Boundary\n(Tool Wear vs Torque)',
    xlabel='Tool Wear [min]  [scaled]',
    ylabel='Torque [Nm]  [scaled]')
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
fig.tight_layout()
save(fig, 'svm_rbf_decision_boundary.png')

# ─────────────────────────────────────────────────────────────────────
# 5. RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────
print('\n══ Random Forest ══')
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_leaf=2, random_state=42,
                             n_jobs=-1)
rf.fit(X_train_arr, y_train_sm)

y_pred_rf   = rf.predict(X_test_arr)
y_prob_rf   = rf.predict_proba(X_test_arr)[:, 1]

prec_rf = precision_score(y_test, y_pred_rf)
rec_rf  = recall_score(y_test, y_pred_rf)
f1_rf   = f1_score(y_test, y_pred_rf)
acc_rf  = accuracy_score(y_test, y_pred_rf)

roc_rf, fpr_rf, tpr_rf = plot_roc(
    y_test, y_prob_rf, 'Random Forest', PALETTE[2],
    'Random Forest — ROC Curve',
    'random_forest_roc_curve.png')

plot_prf_bar(prec_rf, rec_rf, f1_rf,
    'Random Forest — Failure Class Metrics', PALETTE[2],
    'random_forest_prf_bar.png')

# (4) Feature importance bar chart
print('  Plotting feature importance…')
importances = rf.feature_importances_

# Clean up feature names for display
display_names = []
for fn in feature_names:
    n = fn.replace(' [K]','').replace(' [rpm]','').replace(' [Nm]','').replace(' [min]','')
    display_names.append(n)

sorted_idx   = np.argsort(importances)
sorted_imp   = importances[sorted_idx]
sorted_names = [display_names[i] for i in sorted_idx]

fig = dark_fig(9, 6)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL)
colors_bar = [PALETTE[2] if imp > np.median(importances) else ACCENT
              for imp in sorted_imp]
bars = ax.barh(sorted_names, sorted_imp,
               color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.92)
for b, v in zip(bars, sorted_imp):
    ax.text(v + 0.001, b.get_y() + b.get_height()/2,
            f'{v:.4f}', va='center', color=TEXT, fontsize=8.5)
style_ax(ax,
    title='Random Forest — Feature Importance (Mean Gini Reduction)',
    xlabel='Mean Gini Impurity Reduction',
    ylabel='Feature')
ax.set_xlim(0, sorted_imp.max() * 1.18)
fig.tight_layout()
save(fig, 'random_forest_feature_importance.png')

# ─────────────────────────────────────────────────────────────────────
# 6. COMBINED ROC CURVE
# ─────────────────────────────────────────────────────────────────────
print('\n══ Combined ROC Curve ══')
fig = dark_fig(7, 6)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL)

for (fpr, tpr, roc_auc, label, color) in [
    (fpr_lr,  tpr_lr,  roc_lr,  'Logistic Regression', PALETTE[0]),
    (fpr_svm, tpr_svm, roc_svm, 'SVM (RBF)',           PALETTE[1]),
    (fpr_rf,  tpr_rf,  roc_rf,  'Random Forest',       PALETTE[2]),
]:
    ax.plot(fpr, tpr, lw=2.5, color=color,
            label=f'{label}  (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color=color)

ax.plot([0,1],[0,1], color=GRID, lw=1.2, linestyle='--', label='Random')
style_ax(ax,
    title='Combined ROC Curves — All Classifiers',
    xlabel='False Positive Rate',
    ylabel='True Positive Rate')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10,
          loc='lower right')
fig.tight_layout()
save(fig, 'combined_roc_curve.png')

# ─────────────────────────────────────────────────────────────────────
# 7. GROUPED BAR CHART: Recall & F1 across all three models
# ─────────────────────────────────────────────────────────────────────
print('\n══ Grouped Recall/F1 Bar Chart ══')
models  = ['Logistic\nRegression', 'SVM\n(RBF)', 'Random\nForest']
recalls = [rec_lr,  rec_svm,  rec_rf]
f1s     = [f1_lr,   f1_svm,   f1_rf]

x       = np.arange(len(models))
w       = 0.30

fig = dark_fig(8, 5.5)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL)

b1 = ax.bar(x - w/2, recalls, width=w, label='Recall',
            color=PALETTE[1], edgecolor='white', linewidth=0.8, alpha=0.92)
b2 = ax.bar(x + w/2, f1s,     width=w, label='F1-Score',
            color=PALETTE[2], edgecolor='white', linewidth=0.8, alpha=0.92)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
            f'{h:.3f}', ha='center', va='bottom',
            color=TEXT, fontsize=9.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(models, color=TEXT, fontsize=11)
ax.set_ylim(0, 1.15)
style_ax(ax,
    title='Recall & F1-Score Comparison — Failure Class',
    xlabel='Classifier',
    ylabel='Score')
ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=11)
fig.tight_layout()
save(fig, 'combined_recall_f1_bar.png')

# ─────────────────────────────────────────────────────────────────────
# 8. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────
print('\n' + '═'*65)
print('  MODEL PERFORMANCE SUMMARY (Failure Class = Positive)')
print('═'*65)
header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}"
print(header)
print('─'*65)
for name, acc, prec, rec, f1, roc in [
    ('Logistic Regression', acc_lr,  prec_lr,  rec_lr,  f1_lr,  roc_lr),
    ('SVM (RBF)',           acc_svm, prec_svm, rec_svm, f1_svm, roc_svm),
    ('Random Forest',       acc_rf,  prec_rf,  rec_rf,  f1_rf,  roc_rf),
]:
    print(f'{name:<22} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {roc:>6.3f}')
print('═'*65)
print(f'\nAll 14 figures saved to {OUTDIR}')
