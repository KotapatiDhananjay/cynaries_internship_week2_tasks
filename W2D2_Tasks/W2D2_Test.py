import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer


# Load dataset
df = pd.read_csv("W2D2_dataset.csv")


# ------------------------------------------
# BASIC TESTS
# ------------------------------------------

assert len(df) > 0

assert "Age" in df.columns
assert "Income" in df.columns
assert "City" in df.columns
assert "Purchase_Score" in df.columns


# ------------------------------------------
# NUMERIC COLUMNS
# ------------------------------------------

numeric_columns = [
    "Age",
    "Income",
    "Purchase_Score"
]

numeric_data = df[numeric_columns]


# ------------------------------------------
# MEDIAN IMPUTATION TEST
# ------------------------------------------

median_imputer = SimpleImputer(
    strategy="median"
)

median_data = median_imputer.fit_transform(
    numeric_data
)

assert median_data.shape == numeric_data.shape

assert not pd.isnull(median_data).any()


# ------------------------------------------
# MEAN IMPUTATION TEST
# ------------------------------------------

mean_imputer = SimpleImputer(
    strategy="mean"
)

mean_data = mean_imputer.fit_transform(
    numeric_data
)

assert mean_data.shape == numeric_data.shape

assert not pd.isnull(mean_data).any()


# ------------------------------------------
# KNN IMPUTATION TEST
# ------------------------------------------

knn_imputer = KNNImputer(
    n_neighbors=3
)

knn_data = knn_imputer.fit_transform(
    numeric_data
)

assert knn_data.shape == numeric_data.shape

assert not pd.isnull(knn_data).any()


# ------------------------------------------
# CATEGORICAL IMPUTATION TEST
# ------------------------------------------

mode_imputer = SimpleImputer(
    strategy="most_frequent"
)

city_data = mode_imputer.fit_transform(
    df[["City"]]
)

assert not pd.isnull(city_data).any()


print("All W2D2 tests passed successfully!")