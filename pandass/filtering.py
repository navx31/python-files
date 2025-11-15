import pandas as pd

df=pd.read_csv('people.csv')
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("data types:")
print(df.dtypes)
print("first 5 rows:")
print(df.head())

print("Filtering people older than 30:")
filtered_df = df[df['Age'] > 30]
print(filtered_df)  
print("Filtering people with performance score above 85:")
high_performers = df[df['performance_score'] > 85]
print(high_performers)
print("Filtering people from Mumbai:")
mumbai_residents = df[df['City'] == 'Mumbai']
print(mumbai_residents)
print("Filtering people with salary between 50000 and 60000:")
salary_range = df[(df['salary'] >= 50000) & (df['salary'] <= 60000)]
print(salary_range) 
