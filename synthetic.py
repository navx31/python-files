import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('synthetic_dataset.csv')  
df.interpolate(method='linear', inplace=True)
df.fillna({
    'Category': df['Category'].mode()[0],
    'Stock': df['Stock'].mode()[0]

}, inplace=True)
"""print(df.isnull().sum())

print(df.tail(10))"""

print(df.columns)
numeric_columns=['Price', 'Rating',  'Discount']
for col in numeric_columns:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col], orient='h')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
for col in ['Category', 'Stock']:
    sns.countplot(x=df["Category"])
    sns.countplot(x=df["Stock"])
    plt.figure(figsize=(6,4))
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True,)
plt.title('Correlation Matrix')
plt.show()