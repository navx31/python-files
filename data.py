import pandas as pd
import matplotlib.pyplot as plt

data={
    "Name":["Raj","Vivek","Ankita","Riya","Sahil","Neha","Arjun","Pooja","Rahul","Simran","Karan","Tanya","Amit","Sneha","Rohit","Maya","Vikram","Isha","Aditya","Nina","Sameer","Kavya","Dev","Meera","Raghav","Anjali","Tarun","Diya","Kunal","Sana","Arnav","Rhea","Yash","Kiara","Kabir","Aarav","Anaya","Dhruv","Myra","Vivaan","Zara","Shaurya","Aisha","Reyansh","Nidhi","Atharv","Pallavi","Ishaan","Suhana","Ritvik"],
    "Age":[28,34,22,29,31,27,30,26,33,25,32,24,35,28,29,23,31,27,30,26,34,22,28,25,33,24,32,29,31,27,30,26,35,23,28,32,24,29,31,27,30,26,34,22,28,25,33,24,30,23],
    "City":["Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Delhi","Mumbai","Bangalore","Chennai","Kolkata","Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","Pune","Ahmedabad","Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad","delhi"], 
    "salary":[50000,60000,45000,52000,58000,47000,62000,48000,55000,53000,61000,49000,64000,51000,59000,46000,63000,50000,60000,45000,52000,58000,47000,62000,48000,55000,53000,61000,49000,64000,51000,59000,46000,63000,50000,60000,45000,52000,58000,47000,62000,48000,55000,53000,61000,49000,64000,51000,59000,45000],         
    "performance_score":[81,89,77,85,80,92,85,90,78,88,92,80,95,83,87,89,91,82,94,86,90,79,93,84,88,83,90,86,91,79,94,82,88,87,93,78,89,85,90,80,95,84,91,79,86,88,92,81, 90,77],  
    
}
df=pd.DataFrame(data)
df.to_csv('people_data.csv', index=False)
print("CSV file 'people_data.csv' created successfully.")
pd.value_counts(df['Name'])
pd.value_counts(df['City'])
pd.value_counts(df['Age'])
