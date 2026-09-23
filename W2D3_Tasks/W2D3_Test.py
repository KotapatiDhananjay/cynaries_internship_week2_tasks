import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE


# ------------------------------------------
# CREATE DATASET
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


# ------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------
# SMOTE
# ------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ------------------------------------------
# TESTS
# ------------------------------------------

assert len(X_train_smote) == len(y_train_smote)

assert len(X_train_smote) > len(X_train)

class_counts = pd.Series(
    y_train_smote
).value_counts()

assert class_counts.iloc[0] == class_counts.iloc[1]

assert len(X_test) > 0

assert len(y_test) > 0

print("All W2D3 SMOTE tests passed successfully!")