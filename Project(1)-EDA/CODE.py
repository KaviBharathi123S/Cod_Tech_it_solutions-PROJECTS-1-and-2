                     #STEP - 2
                     #LOAD THE DATA
# Step 1: Import core libraries
import pandas as pd   # For loading and manipulating tabular data
import numpy as np    # For numerical operations (used later)

# Step 2: Load the dataset from your local downloaded file
# Update the path/filename to match what you downloaded (e.g., "2019.csv")
df = pd.read_csv("C:/Users/Admin/Desktop/Project(1)-EDA/world Happiness Report dataset/2019.csv")

# Step 3: Preview the first 5 rows
df.head()

# Step 4: Preview the last 5 rows too — good practice to check the file loaded fully
df.tail()
                          #STEP - 3
                          #DATA CLEANING
# Step 1: Standardize column names — lowercase, no spaces, no special characters
df.columns = (
    df.columns
    .str.strip()                # remove hidden leading/trailing spaces
    .str.lower()                # convert to lowercase
    .str.replace(" ", "_")      # replace spaces with underscores
)

# Step 2: Confirm the new clean column names
print(df.columns.tolist())

# Step 3: Double-check data types are sensible 
# (all scores/metrics should be float, rank should be int, country should be text/object)
print(df.dtypes)

# Step 4: Quick final validation — confirm still no missing values after renaming
print("\nMissing values check:\n", df.isnull().sum())

#                          #STEP - 4                        
#                          #Univariate Analysis
# # Step 1: Import visualization libraries
import matplotlib.pyplot as plt   # Core plotting library
import seaborn as sns             # Built on top of matplotlib, prettier defaults + easier syntax

# # Step 2: Set a consistent visual style for all our plots
# sns.set_style("whitegrid")

# # Step 3: Plot the distribution of the main variable: "score" (happiness score)
# plt.figure(figsize=(8, 5))
# sns.histplot(df["score"], bins=20, kde=True)
# plt.title("Distribution of Happiness Scores")
# plt.xlabel("Happiness Score")
# plt.ylabel("Number of Countries")
# plt.show()

# # Step 4: Plot distributions for all numeric columns at once (efficient overview)
# numeric_cols = df.select_dtypes(include="number").columns.drop("overall_rank")
# df[numeric_cols].hist(figsize=(12, 10), bins=15, edgecolor="black")
# plt.suptitle("Distributions of All Numeric Features")
# plt.tight_layout()
# plt.show()

# # Step 5: Boxplot for "score" — visualize spread and check for outliers
# plt.figure(figsize=(6, 4))
# sns.boxplot(x=df["score"])
# plt.title("Boxplot of Happiness Scores")
# plt.show()

# # Step 6: Top 10 and bottom 10 happiest countries — simple ranking view
# print("Top 10 Happiest Countries:")
# print(df[["country_or_region", "score"]].sort_values(by="score", ascending=False).head(10))

# print("\nBottom 10 Happiest Countries:")
# print(df[["country_or_region", "score"]].sort_values(by="score", ascending=True).head(10))
                        #STEP - 5
                        #Bivariate / Multivariate Analysis
# Step 1: Scatter plot — GDP per capita vs. Happiness Score
# plt.figure(figsize=(8, 5))
# sns.scatterplot(data=df, x="gdp_per_capita", y="score")
# plt.title("GDP per Capita vs. Happiness Score")
# plt.xlabel("GDP per Capita")
# plt.ylabel("Happiness Score")
# plt.show()

# # Step 2: Scatter plot — Social Support vs. Happiness Score
# plt.figure(figsize=(8, 5))
# sns.scatterplot(data=df, x="social_support", y="score")
# plt.title("Social Support vs. Happiness Score")
# plt.xlabel("Social Support")
# plt.ylabel("Happiness Score")
# plt.show()

# # Step 3: Scatter plot — Healthy Life Expectancy vs. Happiness Score
# plt.figure(figsize=(8, 5))
# sns.scatterplot(data=df, x="healthy_life_expectancy", y="score")
# plt.title("Healthy Life Expectancy vs. Happiness Score")
# plt.xlabel("Healthy Life Expectancy")
# plt.ylabel("Happiness Score")
# plt.show()

# # Step 4: Pairplot — visualize ALL numeric relationships at once (multivariate view)
# sns.pairplot(df[["score", "gdp_per_capita", "social_support", "healthy_life_expectancy"]])
# plt.suptitle("Pairwise Relationships Between Key Variables", y=1.02)
# plt.show()
                     #STEP 6 
                     #Correlation Analysis 
# Step 1: Calculate the correlation matrix for all numeric columns
corr_matrix = df.select_dtypes(include="number").drop(columns="overall_rank").corr()

# # Step 2: Print it as a table
# print(corr_matrix)

# # Step 3: Visualize it as a heatmap — much easier to read than raw numbers
# plt.figure(figsize=(9, 7))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap of Happiness Features")
# plt.show()

# # Step 4: Specifically check what correlates most strongly with "score"
# print("\nCorrelation with Happiness Score (sorted):")
# print(corr_matrix["score"].sort_values(ascending=False))
                          #STEP 7 
                          #Outlier Detection
# Step 1: Define a reusable function to detect outliers using the IQR method
# def find_outliers_iqr(data, column):
#     Q1 = data[column].quantile(0.25)   # 25th percentile
#     Q3 = data[column].quantile(0.75)   # 75th percentile
#     IQR = Q3 - Q1                       # Interquartile range
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
#     return outliers, lower_bound, upper_bound

# # Step 2: Check for outliers in "score"
# outliers_score, low, high = find_outliers_iqr(df, "score")
# print(f"Valid range for 'score': {low:.2f} to {high:.2f}")
# print(f"Number of outliers in 'score': {len(outliers_score)}")
# print(outliers_score[["country_or_region", "score"]])

# # Step 3: Check outliers across ALL numeric columns at once
# numeric_cols = df.select_dtypes(include="number").columns.drop("overall_rank")
# for col in numeric_cols:
#     outliers, low, high = find_outliers_iqr(df, col)
#     print(f"\n{col}: {len(outliers)} outlier(s) | valid range: {low:.2f} to {high:.2f}")
#     if len(outliers) > 0:
#         print(outliers[["country_or_region", col]])
                       #STEP - 8
                       #SUMMARY(PROGRAMATICALLY)
# Step 1: Pull key numbers directly from your analysis (don't hardcode — reference your variables)
top_country = df.loc[df["score"].idxmax(), "country_or_region"]
top_score = df["score"].max()
bottom_country = df.loc[df["score"].idxmin(), "country_or_region"]
bottom_score = df["score"].min()
avg_score = df["score"].mean()

strongest_driver = corr_matrix["score"].drop("score").idxmax()
strongest_value = corr_matrix["score"].drop("score").max()
weakest_driver = corr_matrix["score"].drop("score").idxmin()
weakest_value = corr_matrix["score"].drop("score").min()

# Step 2: Print a clean, structured summary
print("="*60)
print("EDA SUMMARY: World Happiness Report")
print("="*60)
print(f"Dataset size: {df.shape[0]} countries, {df.shape[1]} features")
print(f"Average happiness score: {avg_score:.2f}")
print(f"Happiest country: {top_country} ({top_score})")
print(f"Least happy country: {bottom_country} ({bottom_score})")
print(f"Strongest driver of happiness: {strongest_driver} (corr = {strongest_value:.2f})")
print(f"Weakest driver of happiness: {weakest_driver} (corr = {weakest_value:.2f})")
print("="*60)