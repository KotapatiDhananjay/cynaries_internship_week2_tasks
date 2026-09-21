import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
from sklearn.feature_selection import SelectKBest, f_regression


# ==========================================
# W2D1 - FEATURE ENGINEERING & ENCODING
# ==========================================

print("=" * 60)
print("W2D1 - FEATURE ENGINEERING & ENCODING")
print("=" * 60)


# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv("W2D1_dataset.csv")

print("\nOriginal Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------
# 2. LABEL ENCODER
# ------------------------------------------

label_encoder = LabelEncoder()

df["Product_Label"] = label_encoder.fit_transform(df["Product"])

print("\nLabel Encoding:")
print(df[["Product", "Product_Label"]].drop_duplicates())


# ------------------------------------------
# 3. ONE HOT ENCODER
# ------------------------------------------

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

city_encoded = one_hot_encoder.fit_transform(df[["City"]])

city_columns = one_hot_encoder.get_feature_names_out(["City"])

city_encoded_df = pd.DataFrame(
    city_encoded,
    columns=city_columns
)

print("\nOne Hot Encoded City:")
print(city_encoded_df.head())


# ------------------------------------------
# 4. ORDINAL ENCODER
# ------------------------------------------

education_order = [
    ["High School", "Graduate", "Postgraduate"]
]

ordinal_encoder = OrdinalEncoder(
    categories=education_order
)

df["Education_Ordinal"] = ordinal_encoder.fit_transform(
    df[["Education_Level"]]
)

print("\nOrdinal Encoding:")
print(
    df[
        ["Education_Level", "Education_Ordinal"]
    ].drop_duplicates()
)


# ------------------------------------------
# 5. SCALING
# ------------------------------------------

sales = df[["Sales"]]

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

df["Sales_StandardScaler"] = (
    standard_scaler.fit_transform(sales)
)

df["Sales_MinMaxScaler"] = (
    minmax_scaler.fit_transform(sales)
)

df["Sales_RobustScaler"] = (
    robust_scaler.fit_transform(sales)
)


print("\nScaled Sales:")
print(
    df[
        [
            "Sales",
            "Sales_StandardScaler",
            "Sales_MinMaxScaler",
            "Sales_RobustScaler"
        ]
    ].head(10)
)


# ------------------------------------------
# 6. PLOT BEFORE SCALING
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Sales"], bins=8)

plt.title("Sales Distribution Before Scaling")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("W2D1_before_scaling.png")

plt.show()


# ------------------------------------------
# 7. PLOT STANDARD SCALER
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales_StandardScaler"],
    bins=8
)

plt.title("Sales Distribution - StandardScaler")
plt.xlabel("Scaled Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("W2D1_standard_scaler.png")

plt.show()


# ------------------------------------------
# 8. PLOT MIN MAX SCALER
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales_MinMaxScaler"],
    bins=8
)

plt.title("Sales Distribution - MinMaxScaler")
plt.xlabel("Scaled Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("W2D1_minmax_scaler.png")

plt.show()


# ------------------------------------------
# 9. PLOT ROBUST SCALER
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sales_RobustScaler"],
    bins=8
)

plt.title("Sales Distribution - RobustScaler")
plt.xlabel("Scaled Sales")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("W2D1_robust_scaler.png")

plt.show()


# ------------------------------------------
# 10. SELECT KBEST
# ------------------------------------------

feature_data = pd.DataFrame()

feature_data["Customer_ID"] = df["Customer_ID"]
feature_data["Product_Label"] = df["Product_Label"]
feature_data["Education_Ordinal"] = df["Education_Ordinal"]
feature_data["Sales_StandardScaler"] = df[
    "Sales_StandardScaler"
]
feature_data["Sales_MinMaxScaler"] = df[
    "Sales_MinMaxScaler"
]

target = df["Sales"]

selector = SelectKBest(
    score_func=f_regression,
    k=5
)

selected_features = selector.fit_transform(
    feature_data,
    target
)

selected_feature_names = feature_data.columns[
    selector.get_support()
]

print("\nTop 5 Selected Features:")
for feature in selected_feature_names:
    print("-", feature)


# ------------------------------------------
# 11. FEATURE SCORES
# ------------------------------------------

feature_scores = pd.DataFrame({
    "Feature": feature_data.columns,
    "Score": selector.scores_
})

feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False
)

print("\nFeature Scores:")
print(feature_scores)


# ------------------------------------------
# 12. SAVE RESULTS
# ------------------------------------------

df.to_csv(
    "W2D1_encoded_scaled_data.csv",
    index=False
)

feature_scores.to_csv(
    "W2D1_feature_scores.csv",
    index=False
)


# ------------------------------------------
# 13. TRADE-OFFS
# ------------------------------------------

print("\nENCODING TRADE-OFFS")
print("--------------------")

print("""
1. LabelEncoder:
   Converts categories into integer labels.
   Simple, but may create an artificial numerical relationship.

2. OneHotEncoder:
   Creates separate binary columns.
   Avoids false ordering, but increases the number of features.

3. OrdinalEncoder:
   Useful when categories have a natural order.
   The category order must be defined correctly.

SCALING:

1. StandardScaler:
   Centers data around mean 0 with standard deviation 1.
   Sensitive to outliers.

2. MinMaxScaler:
   Scales values to a fixed range, usually 0 to 1.
   Sensitive to outliers.

3. RobustScaler:
   Uses median and interquartile range.
   More resistant to outliers.
""")


print("\nFeature engineering completed successfully!")