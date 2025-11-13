import matplotlib.pyplot as plt

lables=["c","python","java","c++","javascript","R","swift"]
no_of_users=[23,45,40,50,38,29,15]  # in millions
plt.pie(no_of_users,labels=lables,autopct='%1.1f%%',startangle=140,shadow=True,explode=(0.1,0,0,0,0,0,0),colors=['gold','yellowgreen','lightcoral','lightskyblue','cyan','magenta','lightgreen'])
plt.title('Popularity of Programming Languages')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()