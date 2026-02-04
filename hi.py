import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('people-100.csv')
df_dup=df.copy()
df_dup.drop(columns=['Phone','First Name'],inplace=True)
print(df_dup.head())
le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'],inplace=True)
print(df_dup.head())
