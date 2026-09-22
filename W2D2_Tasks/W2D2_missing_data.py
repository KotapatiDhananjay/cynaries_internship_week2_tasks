import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer


# ==========================================
# W2D2 - MISSING DATA & IMPUTATION
# ==========================================

print("=" * 60)
print("W2D2 - MISSING DATA & IMPUTATION")
print("=" * 60)


# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv("W2D2_dataset.csv")

print("\nOriginal Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Percentage:")
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)


# ------------------------------------------
# 3. VISUALIZE MISSING VALUES
# ------------------------------------------

plt.figure(figsize=(8, 5))

missing_counts = df.isnull().sum()

missing_counts.plot(kind="bar")

plt.title("Missing Values Before Imputation")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("W2D2_before_imputation.png")

plt.show()


# ------------------------------------------
# 4. NUMERIC COLUMNS
# ------------------------------------------

numeric_columns = [
    "Age",
    "Income",
    "Purchase_Score"
]

numeric_data = df[numeric_columns]


# ------------------------------------------
# 5. MEAN IMPUTATION
# ------------------------------------------

mean_imputer = SimpleImputer(
    strategy="mean"
)

mean_data = mean_imputer.fit_transform(
    numeric_data
)

mean_df = pd.DataFrame(
    mean_data,
    columns=numeric_columns
)

print("\nMean Imputation:")
print(mean_df.head(10))


# ------------------------------------------
# 6. MEDIAN IMPUTATION
# ------------------------------------------

median_imputer = SimpleImputer(
    strategy="median"
)

median_data = median_imputer.fit_transform(
    numeric_data
)

median_df = pd.DataFrame(
    median_data,
    columns=numeric_columns
)

print("\nMedian Imputation:")
print(median_df.head(10))


# ------------------------------------------
# 7. MISSING INDICATOR
# ------------------------------------------

df["Age_was_missing"] = df["Age"].isnull().astype(int)

df["Income_was_missing"] = (
    df["Income"].isnull().astype(int)
)

df["Purchase_Score_was_missing"] = (
    df["Purchase_Score"].isnull().astype(int)
)

print("\nMissing Indicators:")
print(
    df[
        [
            "Age_was_missing",
            "Income_was_missing",
            "Purchase_Score_was_missing"
        ]
    ].head(10)
)


# ------------------------------------------
# 8. KNN IMPUTATION
# ------------------------------------------

knn_imputer = KNNImputer(
    n_neighbors=3
)

knn_data = knn_imputer.fit_transform(
    numeric_data
)

knn_df = pd.DataFrame(
    knn_data,
    columns=numeric_columns
)

print("\nKNN Imputation:")
print(knn_df.head(10))


# ------------------------------------------
# 9. CATEGORICAL IMPUTATION
# ------------------------------------------

mode_imputer = SimpleImputer(
    strategy="most_frequent"
)

df[["City"]] = mode_imputer.fit_transform(
    df[["City"]]
)

print("\nCity After Mode Imputation:")
print(df["City"].head(10))


# ------------------------------------------
# 10. USE MEDIAN VALUES IN FINAL DATASET
# ------------------------------------------

df["Age"] = median_df["Age"]
df["Income"] = median_df["Income"]
df["Purchase_Score"] = median_df["Purchase_Score"]


# ------------------------------------------
# 11. CHECK AFTER IMPUTATION
# ------------------------------------------

print("\nMissing Values After Imputation:")
print(df.isnull().sum())


# ------------------------------------------
# 12. VISUALIZE AFTER IMPUTATION
# ------------------------------------------

plt.figure(figsize=(8, 5))

after_missing = df.isnull().sum()

after_missing.plot(kind="bar")

plt.title("Missing Values After Imputation")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("W2D2_after_imputation.png")

plt.show()


# ------------------------------------------
# 13. SAVE CLEANED DATA
# ------------------------------------------

df.to_csv(
    "W2D2_cleaned_data.csv",
    index=False
)


# ------------------------------------------
# 14. SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("IMPUTATION SUMMARY")
print("=" * 60)

print("""
1. Mean Imputation:
   Replaces missing numeric values with the mean.

2. Median Imputation:
   Replaces missing numeric values with the median.
   It is more robust to outliers.

3. Mode Imputation:
   Replaces missing categorical values with the
   most frequent category.

4. KNN Imputation:
   Estimates missing values using similar rows.

5. Missing Indicator:
   Creates a binary column showing whether the
   original value was missing.
""")


print("\nFinal Dataset:")
print(df.head(10))

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nW2D2 completed successfully!")