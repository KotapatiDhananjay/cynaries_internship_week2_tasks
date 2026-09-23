import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ==========================================
# W2D3 - HANDLING IMBALANCED DATA USING SMOTE
# ==========================================

print("=" * 60)
print("W2D3 - HANDLING IMBALANCED DATA USING SMOTE")
print("=" * 60)


# ------------------------------------------
# 1. CREATE IMBALANCED DATASET
# ------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)

columns = [
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4",
    "Feature_5"
]

df = pd.DataFrame(X, columns=columns)
df["Target"] = y

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution Before SMOTE:")
print(df["Target"].value_counts())


# ------------------------------------------
# 2. SAVE ORIGINAL DATASET
# ------------------------------------------

df.to_csv(
    "W2D3_imbalanced_dataset.csv",
    index=False
)


# ------------------------------------------
# 3. VISUALIZE BEFORE SMOTE
# ------------------------------------------

plt.figure(figsize=(7, 5))

df["Target"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Class Distribution Before SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("W2D3_before_SMOTE.png")

plt.show()


# ------------------------------------------
# 4. SPLIT DATA
# ------------------------------------------

X = df[columns]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Class Distribution Before SMOTE:")
print(y_train.value_counts())


# ------------------------------------------
# 5. APPLY SMOTE
# ------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nTraining Class Distribution After SMOTE:")
print(y_train_smote.value_counts())


# ------------------------------------------
# 6. VISUALIZE AFTER SMOTE
# ------------------------------------------

plt.figure(figsize=(7, 5))

y_train_smote.value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("W2D3_after_SMOTE.png")

plt.show()


# ------------------------------------------
# 7. TRAIN MODEL
# ------------------------------------------

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train_smote,
    y_train_smote
)


# ------------------------------------------
# 8. MAKE PREDICTIONS
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# 9. MODEL EVALUATION
# ------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

print("\nMODEL PERFORMANCE")
print("-----------------")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ------------------------------------------
# 10. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------
# 11. CLASSIFICATION REPORT
# ------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "W2D3_model_results.csv",
    index=False
)


# ------------------------------------------
# 13. SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("SMOTE SUMMARY")
print("=" * 60)

print("""
SMOTE stands for Synthetic Minority Over-sampling Technique.

It creates synthetic samples for the minority class.

SMOTE is applied only to the training data.

The test data is kept unchanged so that model
evaluation remains realistic.

The goal is to reduce class imbalance and help
the model learn the minority class better.
""")


print("\nW2D3 completed successfully!")