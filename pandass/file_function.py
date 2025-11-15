import pandas as pd

df=pd.read_csv('people.csv')
print("Display 3 rows from CSV file:")
print(df.head(3))

print("Display 3 rows form last:")
print(df.tail(3))

print("Display DataFrame info:")
print(df.info())
print("Display summary statistics:")
print(df.describe()) 