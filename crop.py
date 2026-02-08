import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('crop_yield.csv')
print(df.columns)
df_dup=df.copy()

ss=StandardScaler()
df_dup['Temperature'] = ss.fit_transform(df_dup[['Temperature (°C)']])
df_dup['Rainfall'] = ss.fit_transform(df_dup[['Rainfall (mm)']])
df_dup['Humidity'] = ss.fit_transform(df_dup[['Humidity (%)']])
print(df_dup.head())

MM=MinMaxScaler()
df_dup['Temperature'] = MM.fit_transform(df_dup[['Temperature (°C)']])
df_dup['Rainfall'] = MM.fit_transform(df_dup[['Rainfall (mm)']])
df_dup['Humidity'] = MM.fit_transform(df_dup[['Humidity (%)']])
print(df_dup.head())

on=OneHotEncoder()
df_dup = pd.get_dummies(df_dup, columns=['Soil Type','Weather Condition','Crop Type'], dtype=int)
print(df_dup.columns) 

X = df_dup.drop('Yield (tons/hectare)', axis=1)
y = df_dup['Yield (tons/hectare)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("x train is :")
print(X_train)
print("x test is :")
print(X_test)
print("y train is :")
print(y_train)
print("y test is :")
print(y_test)  

