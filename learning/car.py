import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('car.csv')
df_dup=df.copy()
"""print(df_dup.head())
print(df_dup['transmission'].value_counts())"""
le = LabelEncoder()
df_dup['transmission'] = le.fit_transform(df_dup['transmission'])
print(df_dup.head())
df_dup = pd.get_dummies(df_dup, columns=['fuel','seller_type','owner'], dtype=int)
print(df_dup.columns)
print(df_dup.info())
print(df_dup['name'].unique())
