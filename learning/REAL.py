import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

#load dataset 
df=pd.read_csv("real_estate.csv")
df_dup=df.copy()
print(df.columns)
df_dup.drop(columns=["ID","Garage_Size","Location_Score","Distance_to_Center"],inplace=True,axis=1)
print(df_dup.columns)

num_cols = ['Square_Feet','Num_Bedrooms','Num_Bathrooms','Num_Floors','Year_Built']
mm = MinMaxScaler()

# split
X = df_dup.drop('Price', axis=1)
y = df_dup['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# fit on training numeric columns, transform both
mm.fit(X_train[num_cols])
X_train[num_cols] = mm.transform(X_train[num_cols])
X_test[num_cols]  = mm.transform(X_test[num_cols])

# train model on X_train
model = LinearRegression()
model.fit(X_train, y_train)

#make predictions
y_pred=model.predict(X_test)
#evaluate the mode
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R-squared:",r2)
print("Mean Absolute Error:",mae)


#user input for prediction
square_feet=float(input("Enter the square footage of the house: "))
num_bedrooms=float(input("Enter the number of bedrooms: "))
num_bathrooms=float(input("Enter the number of bathrooms: "))
year_built=float(input("Enter the year the house was built: "))
num_floors=float(input("Enter the number of floors: "))
has_garden=float(input("Does the house have a garden? (1 for yes, 0 for no): "))
has_pool=float(input("Does the house have a pool? (1 for yes, 0 for no): "))

#combine raw inputs into one row (same column order used for training)
user_row = pd.DataFrame([[square_feet, num_bedrooms, num_bathrooms, num_floors, year_built, has_garden, has_pool]],
                        columns=X.columns)
user_row[num_cols] = mm.transform(user_row[num_cols])
predicted_price = model.predict(user_row)[0]
print(f"The predicted price of the house is: ${predicted_price:,.2f}")







