import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# W2D4 - TRAIN/TEST SPLIT & CROSS-VALIDATION
# ==========================================

print("=" * 60)
print("W2D4 - TRAIN/TEST SPLIT & CROSS-VALIDATION")
print("=" * 60)


# ------------------------------------------
# 1. CREATE DATASET
# ------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

feature_names = [
    "Feature_1",
    "Feature_2",
    "Feature_3",
    "Feature_4",
    "Feature_5"
]

df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Target"].value_counts())


# ------------------------------------------
# 2. TRAIN / TEST SPLIT
# ------------------------------------------

X = df[feature_names]
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


# ------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------

model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)


# ------------------------------------------
# 4. TEST MODEL
# ------------------------------------------

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nTest Accuracy:")
print(f"{test_accuracy:.4f}")


# ------------------------------------------
# 5. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------
# 6. CLASSIFICATION REPORT
# ------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------
# 7. 5-FOLD CROSS-VALIDATION
# ------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("\nMean Cross-Validation Accuracy:")
print(f"{cv_scores.mean():.4f}")

print("\nStandard Deviation:")
print(f"{cv_scores.std():.4f}")


# ------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Test Accuracy",
        "CV Fold 1",
        "CV Fold 2",
        "CV Fold 3",
        "CV Fold 4",
        "CV Fold 5",
        "Mean CV Accuracy",
        "CV Standard Deviation"
    ],
    "Score": [
        test_accuracy,
        cv_scores[0],
        cv_scores[1],
        cv_scores[2],
        cv_scores[3],
        cv_scores[4],
        cv_scores.mean(),
        cv_scores.std()
    ]
})

results.to_csv(
    "W2D4_cv_results.csv",
    index=False
)


# ------------------------------------------
# 9. VISUALIZE CV SCORES
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    range(1, 6),
    cv_scores
)

plt.axhline(
    cv_scores.mean(),
    linestyle="--",
    label="Mean CV Accuracy"
)

plt.title("5-Fold Cross-Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.xticks(range(1, 6))
plt.legend()

plt.tight_layout()

plt.savefig(
    "W2D4_cross_validation.png"
)

plt.show()


# ------------------------------------------
# 10. SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("W2D4 SUMMARY")
print("=" * 60)

print("""
Train/Test Split:
The dataset is divided into training and testing data.

The model learns from the training data and is
evaluated on the unseen test data.

Cross-Validation:
The training data is divided into multiple folds.
The model is trained and validated multiple times.

5-Fold Cross-Validation provides a more reliable
estimate of model performance.
""")


print("\nW2D4 completed successfully!")