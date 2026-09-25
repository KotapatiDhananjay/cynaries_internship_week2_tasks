import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==========================================
# W2D5 - END-TO-END PREPROCESSING PIPELINE
# ==========================================

print("=" * 60)
print("W2D5 - TITANIC PREPROCESSING PIPELINE")
print("=" * 60)


# ------------------------------------------
# 1. LOAD TITANIC DATASET
# ------------------------------------------

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------
# 2. BASIC EDA
# ------------------------------------------

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)


# ------------------------------------------
# 3. VISUALIZE SURVIVAL
# ------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Survived"
)

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig("W2D5_survival_distribution.png")
plt.show()


# ------------------------------------------
# 4. SELECT FEATURES AND TARGET
# ------------------------------------------

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"

X = df[features]
y = df[target]


# ------------------------------------------
# 5. DEFINE COLUMN TYPES
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
# 6. NUMERIC PIPELINE
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


# ------------------------------------------
# 7. CATEGORICAL PIPELINE
# ------------------------------------------

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
# 8. COLUMN TRANSFORMER
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
# 9. FIT AND TRANSFORM DATA
# ------------------------------------------

X_processed = preprocessor.fit_transform(X)


print("\nProcessed Data Shape:")
print(X_processed.shape)


# ------------------------------------------
# 10. GET FEATURE NAMES
# ------------------------------------------

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print("\nProcessed Feature Names:")
print(feature_names)


# ------------------------------------------
# 11. CREATE ML-READY DATAFRAME
# ------------------------------------------

X_processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

X_processed_df["Survived"] = y.values


# ------------------------------------------
# 12. CHECK MISSING VALUES
# ------------------------------------------

print("\nMissing Values After Processing:")
print(X_processed_df.isnull().sum().sum())


# ------------------------------------------
# 13. SAVE ML-READY DATA
# ------------------------------------------

X_processed_df.to_csv(
    "W2D5_titanic_ml_ready.csv",
    index=False
)


# ------------------------------------------
# 14. SAVE SUMMARY
# ------------------------------------------

summary = pd.DataFrame({
    "Property": [
        "Original Rows",
        "Original Columns",
        "Processed Rows",
        "Processed Columns",
        "Missing Values After Processing"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        X_processed_df.shape[0],
        X_processed_df.shape[1],
        X_processed_df.isnull().sum().sum()
    ]
})

summary.to_csv(
    "W2D5_pipeline_summary.csv",
    index=False
)


# ------------------------------------------
# 15. FINAL SUMMARY
# ------------------------------------------

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)

print("""
1. Loaded Titanic dataset
2. Performed basic EDA
3. Checked missing values
4. Imputed numerical values using median
5. Imputed categorical values using most frequent value
6. Encoded categorical features using OneHotEncoder
7. Scaled numerical features using StandardScaler
8. Used ColumnTransformer for preprocessing
9. Exported ML-ready features
""")


print("\nW2D5 completed successfully!")