import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv('people_data.csv')
plt.bar(df['City'].value_counts().index, df['City'].value_counts().values)
plt.title('Number of People in Each City')
plt.xlabel('City')
plt.ylabel('Number of People')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.hist(df['Age'], bins=10, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.tight_layout()
plt.show()

plt.scatter(df['Age'], df['salary'])
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.tight_layout()
plt.show()

plt.boxplot(df['performance_score'])
plt.title('Performance Score Distribution')
plt.ylabel('Performance Score')
plt.tight_layout()
plt.show()  

plt.pie(df['City'].value_counts().values, labels=df['City'].value_counts().index, autopct='%1.1f%%')
plt.title('City Distribution')
plt.tight_layout()
plt.show()  

plt.plot(df['Name'], df['salary'], marker='o')
plt.title('Salary by Name')
plt.xlabel('Name')
plt.ylabel('Salary')
plt.xticks(rotation=90) 
plt.tight_layout()
plt.show()


