import matplotlib.pyplot as plt

county=["canada","usa","mexico","india","argentina"]
population=[38,331,128,1393,45]  # in millions

plt.bar(county,population,color=['red','blue','green','orange','purple'])
plt.title('Population of Different Countries')
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.ylim(0,1500)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()