import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('car.csv')
df_dup=df.copy()

df_dup.drop(['name','year','fuel','seller_type','transmission','owner'],axis=1,inplace=True)
X=df_dup.drop('selling_price',axis=1)
y=df_dup['selling_price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)

km = input("enter km driven: ")
try:
    km_val = float(km)
except ValueError:
    raise ValueError("Invalid km input, enter a numeric value")
# use a DataFrame with the same column name as training features
X_new = pd.DataFrame({"km_driven": [km_val]})
selling_price = lr.predict(X_new)
print(f"The predicted selling price for a car with {km_val} km driven is: ${selling_price[0]:,.2f}")

y_pred=lr.predict(X_test)



