import matplotlib.pyplot as plt

data = [12, 15, 13, 17, 19, 21, 22, 18, 16, 14, 20, 23, 25, 24, 22, 21, 19, 18, 17, 16]
plt.hist(data, bins=5, color='orange', edgecolor='black')
plt.title('Histogram of Data Distribution')
plt.xlabel('Value Ranges')
plt.ylabel('Frequency')
plt.xticks(range(10, 31, 2))
plt.yticks(range(0, 7, 1))
plt.xlim(10, 30)
plt.ylim(0, 6)
plt.grid(True)
plt.grid(axis='y', alpha=0.75)

plt.show()