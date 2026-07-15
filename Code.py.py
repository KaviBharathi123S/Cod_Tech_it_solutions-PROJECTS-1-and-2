# Step 1: Import core libraries
import pandas as pd
import numpy as np

# Step 2: Load the dataset from your local file
df = pd.read_csv(r"C:\Users\Admin\Desktop\Project(2)-Titatnic Passenger Survival Study\Titanic Dataset\train.csv")

# Step 3: Preview first and last rows
df.head()
df.tail()
                       #STEP - 2
                       #DATA OVERVIEW AND STRUCTURE
# Step 1: Shape of the dataset
# print("Shape:", df.shape)

# # Step 2: Column names
# print("\nColumns:", df.columns.tolist())

# # Step 3: Data types and non-null counts
# df.info()

# # Step 4: Summary statistics for numeric columns
# print(df.describe())

# # Step 5: Missing values per column
# print("\nMissing values:\n", df.isnull().sum())

# # Step 6: Duplicate rows
# print("\nDuplicate rows:", df.duplicated().sum())
                        #STEP - 3
                        #DATA CLEANING
# Step 1: Standardize column names (lowercase, no spaces)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(df.columns.tolist())

# Step 2: Fill missing 'age' with the median
df["age"] = df["age"].fillna(df["age"].median())

# Step 3: Fill missing 'embarked' with the most common port (mode)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Step 4: Convert 'cabin' into a binary flag instead of filling it directly
df["has_cabin"] = df["cabin"].notnull().astype(int)
df = df.drop(columns=["cabin"])

# Step 5: Drop columns not useful for analysis
df = df.drop(columns=["passengerid", "ticket"])

# Step 6: Remove duplicates (none expected, but good practice to check again)
df = df.drop_duplicates()

# Step 7: Reset index after cleaning
df = df.reset_index(drop=True)

# Step 8: Final verification
print("\nRemaining missing values:\n", df.isnull().sum())
print("\nFinal shape:", df.shape)
                          #STEP - 4
                          #UNIVARIATE ANALYSIS
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# # --- Numeric Variables ---
# plt.figure(figsize=(8, 5))
# sns.histplot(df["age"], bins=30, kde=True)
# plt.title("Distribution of Passenger Age")
# plt.xlabel("Age")
# plt.show()

# plt.figure(figsize=(8, 5))
# sns.histplot(df["fare"], bins=40, kde=True)
# plt.title("Distribution of Ticket Fare")
# plt.xlabel("Fare")
# plt.show()

# # --- Categorical Variables ---
# plt.figure(figsize=(6, 4))
# sns.countplot(x="survived", data=df)
# plt.title("Survival Counts (0 = Died, 1 = Survived)")
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.countplot(x="pclass", data=df)
# plt.title("Passenger Class Distribution")
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.countplot(x="sex", data=df)
# plt.title("Gender Distribution")
# plt.show()

# plt.figure(figsize=(6, 4))
# sns.countplot(x="embarked", data=df)
# plt.title("Port of Embarkation Distribution")
# plt.show()
                                   #STEP - 5
                                   #BIVARIATE ANALYSIS
# Step 1: Survival rate by gender
# plt.figure(figsize=(6, 4))
# sns.barplot(x="sex", y="survived", data=df)
# plt.title("Survival Rate by Gender")
# plt.ylabel("Survival Rate")
# plt.show()

# print("Survival rate by gender:\n", df.groupby("sex")["survived"].mean())

# # Step 2: Survival rate by passenger class
# plt.figure(figsize=(6, 4))
# sns.barplot(x="pclass", y="survived", data=df)
# plt.title("Survival Rate by Passenger Class")
# plt.ylabel("Survival Rate")
# plt.show()

# print("\nSurvival rate by class:\n", df.groupby("pclass")["survived"].mean())

# # Step 3: Survival rate by class AND gender combined
# plt.figure(figsize=(8, 5))
# sns.barplot(x="pclass", y="survived", hue="sex", data=df)
# plt.title("Survival Rate by Class and Gender")
# plt.ylabel("Survival Rate")
# plt.show()

# # Step 4: Age distribution split by survival outcome
# plt.figure(figsize=(8, 5))
# sns.histplot(data=df, x="age", hue="survived", kde=True, bins=30, element="step")
# plt.title("Age Distribution: Survived vs. Did Not Survive")
# plt.show()

# # Step 5: Fare vs. Survival (boxplot — did wealthier passengers survive more?)
# plt.figure(figsize=(6, 5))
# sns.boxplot(x="survived", y="fare", data=df)
# plt.title("Fare Distribution by Survival Outcome")
# plt.show()

# # Step 6: Does having a recorded cabin relate to survival?
# plt.figure(figsize=(6, 4))
# sns.barplot(x="has_cabin", y="survived", data=df)
# plt.title("Survival Rate by Cabin Data Availability")
# plt.show()

# print("\nSurvival rate by has_cabin:\n", df.groupby("has_cabin")["survived"].mean())
                                   #STEP-6
                                   #CORRELATION ANALYSIS
# Step 1: Create a temporary copy for correlation purposes
# We encode categorical columns as numbers WITHOUT overwriting our main cleaned df
# df_corr = df.copy()

# # Step 2: Convert 'sex' into numeric (0 = male, 1 = female)
# df_corr["sex"] = df_corr["sex"].map({"male": 0, "female": 1})

# # Step 3: Convert 'embarked' into numeric using one-hot encoding
# # (creates separate 0/1 columns for each port: embarked_C, embarked_Q, embarked_S)
# df_corr = pd.get_dummies(df_corr, columns=["embarked"], drop_first=True)

# # Step 4: Drop 'name' — it's unique text per passenger, not usable for correlation
# df_corr = df_corr.drop(columns=["name"])

# # Step 5: Calculate the correlation matrix
# corr_matrix = df_corr.corr()

# # Step 6: Print correlation specifically with 'survived', sorted
# print("Correlation with Survival (sorted):")
# print(corr_matrix["survived"].sort_values(ascending=False))

# # Step 7: Visualize the full correlation matrix as a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap - Titanic Features")
# plt.show()
                                 #STEP-7
                                 #OUTLIER DETECTION
# Step 1: Reuse the same outlier-detection function from your Happiness project
# def find_outliers_iqr(data, column):
#     Q1 = data[column].quantile(0.25)
#     Q3 = data[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
#     return outliers, lower_bound, upper_bound

# # Step 2: Check outliers in 'fare' specifically (we predicted this would be skewed)
# outliers_fare, low, high = find_outliers_iqr(df, "fare")
# print(f"Valid range for 'fare': {low:.2f} to {high:.2f}")
# print(f"Number of outliers in 'fare': {len(outliers_fare)}")
# print(outliers_fare[["name", "fare", "pclass", "survived"]].sort_values(by="fare", ascending=False).head(10))

# # Step 3: Check outliers in 'age'
# outliers_age, low, high = find_outliers_iqr(df, "age")
# print(f"\nValid range for 'age': {low:.2f} to {high:.2f}")
# print(f"Number of outliers in 'age': {len(outliers_age)}")
# print(outliers_age[["name", "age", "survived"]].sort_values(by="age", ascending=False).head(10))

# # Step 4: Check outliers across SibSp and Parch (family size variables)
# for col in ["sibsp", "parch"]:
#     outliers, low, high = find_outliers_iqr(df, col)
#     print(f"\n{col}: {len(outliers)} outlier(s) | valid range: {low:.2f} to {high:.2f}")
                                  #STEP-8
                                  #A-FEATURE ENGINEERING AND B- MULTI-FACTOR SURVIVAL PROFILE
# --- Part A: Feature Engineering ---

# Step 1: Extract 'title' from the 'name' column (e.g., "Mr.", "Mrs.", "Miss.", "Master.")
# Names follow the pattern "Last, Title. First" — we extract the word between ", " and "."
df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.")

# Step 2: Simplify rare titles into broader, more meaningful buckets
# (there are ~17 unique titles, many with only 1-2 passengers — grouping avoids noise)
rare_titles = ["Dr", "Rev", "Col", "Major", "Capt", "Jonkheer", "Sir", "Countess", "Don", "Lady", "Dona"]
df["title"] = df["title"].replace(rare_titles, "Rare")
df["title"] = df["title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

# Step 3: Create 'family_size' — total people traveling together, including the passenger
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Step 4: Create 'is_alone' — a simple flag for solo travelers
df["is_alone"] = (df["family_size"] == 1).astype(int)

# Step 5: Quick check — see the new columns
print(df[["name", "title", "family_size", "is_alone"]].head(10))


# --- Part B: Multi-Factor Survival Profile ---

# Step 6: Survival rate by title
print("\nSurvival rate by title:")
print(df.groupby("title")["survived"].mean().sort_values(ascending=False))

# Step 7: Survival rate by family size
print("\nSurvival rate by family size:")
print(df.groupby("family_size")["survived"].mean())

# Step 8: Survival rate: alone vs. with family
print("\nSurvival rate by is_alone:")
print(df.groupby("is_alone")["survived"].mean())

# Step 9: THE KEY OUTPUT — combined survival profile by class AND sex
profile = pd.pivot_table(df, values="survived", index="pclass", columns="sex", aggfunc="mean")
print("\nSurvival rate by Class and Sex:")
print(profile)

# Step 10: Visualize this combined profile as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(profile, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Survival Rate by Class and Sex")
plt.show()