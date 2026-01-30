import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('people_data.csv')
print("First 5 rows:")
print(df.head())
print("last 5 rows:")
print(df.tail())

print(df.info())
print(df.describe())
print({df.shape})
print(f'colums name:{df.columns}')

name=df['Name']
print(name)

multi_colums=df[['Name', 'Salary']]
print(multi_colums)
filter= df[(df['Salary']>40000)& (df['Salary']<50000)]
print(filter)
filter2=df[(df['Salary']>55000)| (df['performance_score']>90)]
print(filter2)

df["bonus"]=df['Salary']*0.1
print(df)

df.insert(0,"Employee Id",[10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90,10,10,20,30,40,50,60,70,80,90,23,34,45,46])
print(df)

df.loc[23,"Name"]="manan"
print(df)

df['Salary']=df['Salary']*1.05
print(df)

df.insert(3,"experince",(['Age'] * 10)%100)
print(df)
