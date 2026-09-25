import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ------------------------------------------
# CREATE DATASET
# ------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)


# ------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------
# TEST SPLIT
# ------------------------------------------

assert len(X_train) == 800
assert len(X_test) == 200

assert len(y_train) == 800
assert len(y_test) == 200


# ------------------------------------------
# TRAIN MODEL
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
# TEST CROSS-VALIDATION
# ------------------------------------------

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

assert len(cv_scores) == 5

assert all(
    0 <= score <= 1
    for score in cv_scores
)


print("All W2D4 train/test split and cross-validation tests passed successfully!")