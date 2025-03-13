import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the Oral Cancer Prediction Dataset
df = pd.read_csv("oral_cancer_prediction_dataset.csv")
# 1. Data Cleaning
print("\nDataset Overview:")
print(df.info())  # Check structure
print("\nMissing Values:")
print(df.isnull().sum())  # Check missing values

# Handling missing values (Impute with median for numeric, mode for categorical)
df.fillna(df.median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Detect and Remove Outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df = df[~outlier_condition.any(axis=1)]  # Remove outliers

# Standardizing categorical values (Convert to lowercase and strip spaces)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.lower().str.strip()

# 2. Exploratory Data Analysis (EDA)
# Univariate Analysis
print("\nSummary Statistics:")
print(df.describe())

# Histograms
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Boxplots
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# Bivariate Analysis
# Correlation Matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Matrix")
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.show()

# Scatter Plots for Numerical Relationships
num_pairs = [("age", "cancer_stage"), ("smoking", "risk_factor"), ("alcohol_consumption", "cancer_stage")]
for x_col, y_col in num_pairs:
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x_col], df[y_col], alpha=0.5, color='red')
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

print("EDA Complete!")

# Save cleaned dataset
df.to_csv("cleaned_oral_cancer_dataset.csv", index=False)
print("Cleaned dataset saved as cleaned_oral_cancer_dataset.csv")
