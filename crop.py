import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('crop_yield.csv')
print(df.columns)
numeric_columns=['Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)', 'Yield (tons/hectare)']
for col in numeric_columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

sns.countplot(x=df['Soil Type'])
plt.show()
sns.countplot(x=df['Crop Type'])
plt.show()
sns.countplot(x=df['Weather Condition'])
plt.show()

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col], orient='h')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
   
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""df_cleared=df.copy()
print(df_cleared['Soil Type'].value_counts())
print(df_cleared['Crop Type'].value_counts())
print(df_cleared['Weather Condition'].value_counts())

df_cleared=pd.get_dummies(df_cleared, columns=['Soil Type', 'Crop Type', 'Weather Condition'], drop_first=True)
print(df_cleared.head())"""
