import matplotlib.pyplot as plt

months=[1,2,3,4,5,6,7,8,9,10,11,12]
sales=[250,300,280,350,400,375,390,430,435,410,400,450]

plt.plot(months,sales,marker='o',color='blue')
plt.title('Monthly Sales Data')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True)
plt.xticks(months)
plt.yticks(range(200, 501, 50))

plt.show()