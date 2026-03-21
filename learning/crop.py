import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n Model Performance ")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")  

