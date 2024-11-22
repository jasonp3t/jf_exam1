import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

# Load the data
path = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
df = pd.read_csv(path)

# Streamlit App

# Title and description
st.title("Car Price Analysis")
st.write("""
This application explores the relationship between various car features and their prices.
We will be analyzing and visualizing the dataset to answer the question:
**What are the main characteristics that have the most impact on car price?**
""")

# Display the dataframe preview
st.header("Dataset Preview")
st.dataframe(df.head())

# Data types of the columns
st.subheader("Data Types of Columns")
st.write(df.dtypes)

# Question 1: What is the data type of the column "peak-rpm"?
st.subheader("Question 1: Data Type of 'peak-rpm'")
st.write(f"The data type of 'peak-rpm' is {df['peak-rpm'].dtype}.")

# Question 2: Correlation between specific columns
st.subheader("Question 2: Correlation between 'bore', 'stroke', 'compression-ratio', and 'horsepower'")
correlation = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write(correlation)

# Visualization - Positive Linear Relationship (Engine size vs Price)
st.subheader("Positive Linear Relationship: Engine Size vs Price")
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0, )
st.pyplot()

# Correlation between Engine Size and Price
st.write(f"Correlation between 'engine-size' and 'price': {df[['engine-size', 'price']].corr().iloc[0, 1]}")

# Visualization - Negative Relationship (Highway MPG vs Price)
st.subheader("Negative Linear Relationship: Highway MPG vs Price")
sns.regplot(x="highway-mpg", y="price", data=df)
st.pyplot()

# Correlation between Highway MPG and Price
st.write(f"Correlation between 'highway-mpg' and 'price': {df[['highway-mpg', 'price']].corr().iloc[0, 1]}")

# Visualization - Weak Relationship (Peak RPM vs Price)
st.subheader("Weak Linear Relationship: Peak RPM vs Price")
sns.regplot(x="peak-rpm", y="price", data=df)
st.pyplot()

# Correlation between Peak RPM and Price
st.write(f"Correlation between 'peak-rpm' and 'price': {df[['peak-rpm', 'price']].corr().iloc[0, 1]}")

# Question 3: Correlation between 'stroke' and 'price'
st.subheader("Question 3a: Correlation between 'stroke' and 'price'")
st.write(df[['stroke', 'price']].corr())

# Streamlit visualization for 'stroke' vs 'price'
st.subheader("Question 3b: Stroke vs Price Visualization")
sns.regplot(x="stroke", y="price", data=df)
st.pyplot()

# Categorical Variable Analysis - Body Style vs Price
st.subheader("Body Style vs Price")
sns.boxplot(x="body-style", y="price", data=df)
st.pyplot()

# Engine Location vs Price
st.subheader("Engine Location vs Price")
sns.boxplot(x="engine-location", y="price", data=df)
st.pyplot()

# Drive Wheels vs Price
st.subheader("Drive Wheels vs Price")
sns.boxplot(x="drive-wheels", y="price", data=df)
st.pyplot()

# Descriptive Statistical Analysis
st.subheader("Descriptive Statistical Analysis")
st.write("Basic statistical summary of the dataset:")
st.write(df.describe())

# Grouping - Average price by Body Style
st.subheader("Average Price by Body Style")
df_group_two = df[['body-style', 'price']].groupby('body-style', as_index=False).mean()
st.write(df_group_two)

# Grouping by Drive Wheels and Body Style
st.subheader("Average Price by Drive Wheels and Body Style")
df_group_one = df[['drive-wheels', 'price']].groupby('drive-wheels', as_index=False).mean()
st.write(df_group_one)

# Visualization: Heatmap of Drive Wheels and Body Style vs Price
st.subheader("Heatmap of Drive Wheels and Body Style vs Price")
grouped_test1 = df[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)

# Plot heatmap
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# Labeling heatmap
row_labels = grouped_pivot.columns
col_labels = grouped_pivot.index
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
plt.xticks(rotation=90)

fig.colorbar(im)
st.pyplot()

# Correlation Analysis - Pearson Correlation Coefficients and P-values
st.subheader("Correlation and P-Value Calculations")

# Pearson Correlation and P-Value for 'wheel-base' and 'price'
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'wheel-base' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'horsepower' and 'price'
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'horsepower' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'length' and 'price'
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'length' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'width' and 'price'
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'width' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'curb-weight' and 'price'
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'curb-weight' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'engine-size' and 'price'
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'engine-size' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'bore' and 'price'
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'bore' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'city-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'city-mpg' and 'price': {pearson_coef}, p-value: {p_value}")

# Pearson Correlation and P-Value for 'highway-mpg' and 'price'
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
st.write(f"Pearson Correlation Coefficient between 'highway-mpg' and 'price': {pearson_coef}, p-value: {p_value}")

# Conclusion: Important Variables
st.subheader("Conclusion: Important Variables")
st.write("""
The following variables have the most significant impact on car price:
- Continuous numerical variables: Engine-size, Horsepower, Curb-weight, Length, Width, Wheel-base, Bore
- Categorical variables: Drive-wheels
""")

# Footer
st.write("This app helps explore and visualize the relationships between car features and price. Please feel free to analyze the data further.")
