import pandas as pd 

df1 = pd.DataFrame({
    'key': ['A', 'B', 'C', 'D'],
    'value': [1, 2, 3, 4],
})
df2 = pd.DataFrame({
    'key': ['B', 'C', 'D', 'E'],
    'value': [5, 6, 7, 8],
})

merged_df = pd.merge(df1, df2, on='key', how='inner')
print("Merged DataFrame:")
print(merged_df)    

merged_df_left = pd.merge(df1, df2, on='key', how='left')
print("Left Merged DataFrame:")
print(merged_df_left)   

merged_df_right = pd.merge(df1, df2, on='key', how='right') 
print("Right Merged DataFrame:")
print(merged_df_right)  


merged_df_outer = pd.merge(df1, df2, on='key', how='outer')
print("Outer Merged DataFrame:")
print(merged_df_outer)  