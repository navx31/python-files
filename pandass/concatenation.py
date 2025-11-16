import pandas as pd 

df1=pd.DataFrame({
    'City': ['New York', 'Los Angeles', 'Chicago'],
    'Sales': [2500, 3000, 1500],
})
df2=pd.DataFrame({
    'City': ['Chicago', 'Houston', 'Phoenix'],
    'Sales': [2000, 3500, 4000],
})

df_concat=pd.concat([df1, df2], ignore_index=True)
print("Concatenated DataFrame:")
print(df_concat)