import pandas as pd

data={
     "Name":["Raj","Vivek","Ankita","Riya","Sahil","Neha","Arjun","Pooja"],
     "Age":[28,34,22,29,31,27,30,26],
     "City":["Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad"],
     "salary":[50000,60000,45000,52000,58000,47000,62000,48000],
     "performance_score":[85,90,78,88,92,80,95,83]
    }

df=pd.DataFrame(data)
df.to_csv('people.csv', index=False)
print("CSV file 'people.csv' created successfully.")
df.to_json('people.json', orient='records', lines=True)
print("JSON file 'people.json' created successfully.")
