import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------
# CREATE TEST DATA
# ------------------------------------------

data = {
    "Pclass": [1, 2, 3, 1],
    "Sex": ["male", "female", "male", "female"],
    "Age": [22, None, 35, 28],
    "SibSp": [1, 1, 0, 0],
    "Parch": [0, 0, 0, 1],
    "Fare": [7.25, 71.28, 8.05, 53.10],
    "Embarked": ["S", "C", "S", None]
}

df = pd.DataFrame(data)


# ------------------------------------------
# FEATURES
# ------------------------------------------

numeric_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

categorical_features = [
    "Sex",
    "Embarked"
]


# ------------------------------------------
# PIPELINES
# ------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ------------------------------------------
# COLUMN TRANSFORMER
# ------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ------------------------------------------
# TRANSFORM
# ------------------------------------------

X_processed = preprocessor.fit_transform(df)


# ------------------------------------------
# TESTS
# ------------------------------------------

assert X_processed.shape[0] == 4

assert X_processed.shape[1] > len(numeric_features)

assert not pd.isnull(X_processed).any()

print("All W2D5 preprocessing pipeline tests passed successfully!")