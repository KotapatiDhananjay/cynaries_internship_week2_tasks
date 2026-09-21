import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.feature_selection import SelectKBest


# Load dataset
df = pd.read_csv("W2D1_dataset.csv")

# Basic dataset test
assert len(df) > 0
assert "City" in df.columns
assert "Product" in df.columns
assert "Education_Level" in df.columns
assert "Sales" in df.columns


# LabelEncoder test
label_encoder = LabelEncoder()
encoded_product = label_encoder.fit_transform(df["Product"])

assert len(encoded_product) == len(df)


# OneHotEncoder test
one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_city = one_hot_encoder.fit_transform(
    df[["City"]]
)

assert encoded_city.shape[0] == len(df)


# OrdinalEncoder test
ordinal_encoder = OrdinalEncoder()

encoded_education = ordinal_encoder.fit_transform(
    df[["Education_Level"]]
)

assert encoded_education.shape[0] == len(df)


# Scaling tests
sales = df[["Sales"]]

standard = StandardScaler().fit_transform(sales)
minmax = MinMaxScaler().fit_transform(sales)
robust = RobustScaler().fit_transform(sales)

assert standard.shape == sales.shape
assert minmax.shape == sales.shape
assert robust.shape == sales.shape


# SelectKBest test
selector = SelectKBest(
    k=2
)

selected = selector.fit_transform(
    standard,
    df["Sales"]
)

assert selected.shape[0] == len(df)


print("All W2D1 tests passed successfully!")