import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv('ai4i2020.csv')

# -----------------------------
# 2. Define features and target
# -----------------------------
# Drop ID columns and failure-type columns to avoid data leakage.
# Machine failure is derived from the specific failure indicators:
# TWF, HDF, PWF, OSF, RNF
X = df.drop(columns=['Machine failure', 'UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# -----------------------------
# 3. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            numeric_features
        ),
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]),
            categorical_features
        )
    ]
)

# -----------------------------
# 4. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# -----------------------------
# 5. Train Naive Bayes model
# -----------------------------
model = GaussianNB()
model.fit(X_train_processed, y_train)

# -----------------------------
# 6. Predictions
# -----------------------------
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]

# -----------------------------
# 7. Metrics
# -----------------------------
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
print('Naive Bayes Performance on AI4I 2020 Dataset')
print(f'Precision : {precision:.4f}')
print(f'Recall    : {recall:.4f}')
print(f'F1-score  : {f1:.4f}')
print(f'ROC-AUC   : {roc_auc:.4f}')

# -----------------------------
# 8. Plot metrics bar chart
# -----------------------------
metrics = {
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'ROC-AUC': roc_auc
}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.title('Naive Bayes Performance Metrics')
plt.ylabel('Score')
for i, value in enumerate(metrics.values()):
    plt.text(i, value + 0.02, f'{value:.3f}', ha='center')
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=200)
plt.close()

# -----------------------------
# 9. Plot ROC curve
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title('ROC Curve - Naive Bayes')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=200)
plt.close()

