import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42
CSV_PATH = "ai4i2020.csv"

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Standardize expected column names from AI4I 2020 dataset
column_map = {
    "Type": "Type",
    "Air temperature [K]": "T_air",
    "Process temperature [K]": "T_Process",
    "Rotational speed [rpm]": "Rotational_speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool_wear",
    "Machine failure": "Machine_failure",
}

missing_cols = [c for c in column_map if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in CSV: {missing_cols}")

df = df.rename(columns=column_map)

# -----------------------------
# Feature engineering
# -----------------------------
df["Mechanical_power"] = df["Torque"] * (2 * np.pi) * df["Rotational_speed"] / 60.0
df["Wear_torque_interaction"] = df["Tool_wear"] * df["Torque"]
df["Temperature_differential"] = df["T_Process"] - df["T_air"]

# -----------------------------
# Prepare inputs and target
# -----------------------------
raw_sensor_features = [
    "T_air",
    "T_Process",
    "Rotational_speed",
    "Torque",
    "Tool_wear",
]

derived_features = [
    "Mechanical_power",
    "Wear_torque_interaction",
    "Temperature_differential",
]

categorical_feature = "Type"
target_col = "Machine_failure"

type_dummies = pd.get_dummies(df[categorical_feature], prefix="Type", drop_first=False)

X = pd.concat([df[raw_sensor_features + derived_features], type_dummies], axis=1)
y = df[target_col]

numeric_features = raw_sensor_features + derived_features
all_feature_names = X.columns.tolist()

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# Scale numeric features only
# -----------------------------
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# -----------------------------
# SMOTE on training set only
# -----------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000
    ),
    "SVM (RBF Kernel)": SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
}

results = []
roc_data = {}

# -----------------------------
# Plot helper functions
# -----------------------------
def save_roc_curve(y_true, y_score, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    return fpr, tpr, roc_auc


def save_prf_bar_chart(precision, recall, f1, title, filename):
    metrics = ["Precision", "Recall", "F1-score"]
    values = [precision, recall, f1]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(metrics, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(title)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# -----------------------------
# Train, evaluate, and save figures
# -----------------------------
for model_name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    })

    roc_data[model_name] = {
        "y_prob": y_prob
    }

    stem = (
        model_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
    )

    fpr, tpr, model_auc = save_roc_curve(
        y_test,
        y_prob,
        f"{model_name} - ROC Curve",
        f"{stem}_roc_curve.png"
    )
    roc_data[model_name]["fpr"] = fpr
    roc_data[model_name]["tpr"] = tpr
    roc_data[model_name]["auc"] = model_auc

    save_prf_bar_chart(
        precision,
        recall,
        f1,
        f"{model_name} - Failure Class Metrics",
        f"{stem}_failure_class_metrics.png"
    )

    if model_name == "Logistic Regression":
        x_vals = np.linspace(-6, 6, 300)
        sigmoid = 1 / (1 + np.exp(-x_vals))

        x_points = np.linspace(-6, 6, 25)
        y_points = 1 / (1 + np.exp(-x_points))

        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, sigmoid, color="blue", linewidth=2, label="Sigmoid curve")
        plt.scatter(x_points, y_points, color="red", s=30, label="Data points")
        plt.xlabel("X")
        plt.ylabel("sigmoid(X)")
        plt.title("Logistic Regression Sigmoid Function")
        plt.legend()
        plt.tight_layout()
        plt.savefig("logistic_regression_sigmoid.png", bbox_inches="tight")
        plt.close()

    elif model_name == "SVM (RBF Kernel)":
        two_features = ["Tool_wear", "Torque"]

        X2 = df[two_features].copy()
        y2 = df[target_col].copy()

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y2
        )

        scaler_2d = StandardScaler()
        X2_train_scaled = scaler_2d.fit_transform(X2_train)
        X2_test_scaled = scaler_2d.transform(X2_test)

        smote_2d = SMOTE(random_state=RANDOM_STATE)
        X2_train_res, y2_train_res = smote_2d.fit_resample(X2_train_scaled, y2_train)

        svm_2d = SVC(kernel="rbf", random_state=RANDOM_STATE)
        svm_2d.fit(X2_train_res, y2_train_res)

        x_min, x_max = X2_test_scaled[:, 0].min() - 1.0, X2_test_scaled[:, 0].max() + 1.0
        y_min, y_max = X2_test_scaled[:, 1].min() - 1.0, X2_test_scaled[:, 1].max() + 1.0
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 400),
            np.linspace(y_min, y_max, 400)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = svm_2d.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
        plt.scatter(
            X2_test_scaled[:, 0],
            X2_test_scaled[:, 1],
            c=y2_test,
            cmap="coolwarm",
            edgecolor="k",
            s=35
        )
        plt.xlabel("Tool wear (scaled)")
        plt.ylabel("Torque (scaled)")
        plt.title("SVM with RBF Kernel - Decision Boundary")
        plt.tight_layout()
        plt.savefig("svm_rbf_decision_boundary.png", bbox_inches="tight")
        plt.close()

    elif model_name == "Random Forest":
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": all_feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(9, 7))
        sns.barplot(data=fi_df, x="Importance", y="Feature", orient="h")
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Mean decrease in impurity")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("random_forest_feature_importance.png", bbox_inches="tight")
        plt.close()

# -----------------------------
# Combined ROC-AUC curve
# -----------------------------
plt.figure(figsize=(8, 6))
for model_name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], lw=2, label=f"{model_name} (AUC = {data['auc']:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("combined_roc_auc_curve.png", bbox_inches="tight")
plt.close()

# -----------------------------
# Grouped bar chart for Recall and F1-score
# -----------------------------
results_df = pd.DataFrame(results)

x = np.arange(len(results_df))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width / 2, results_df["Recall"], width, label="Recall")
plt.bar(x + width / 2, results_df["F1-score"], width, label="F1-score")

plt.xticks(x, results_df["Model"], rotation=15)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Recall and F1-score Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("model_recall_f1_comparison.png", bbox_inches="tight")
plt.close()

# -----------------------------
# Summary table
# -----------------------------
results_df = results_df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
