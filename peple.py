import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df= pd.read_csv('people_data.csv')
df_dup=df.copy()
df_dup.drop(['Name','Age','City'],axis=1,inplace=True)
X=df_dup[['performance_score']]
y=df_dup[['Salary']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
slr=LinearRegression()
slr.fit(X,y)
perfoemance_score= float(input("Enter the performance score: "))
predicted_salary=slr.predict([[perfoemance_score]])
print(f"The predicted salary for a performance score of {perfoemance_score} is: ${predicted_salary[0][0]:,.2f}")
print("\n--- Model Performance ---")
y_pred=slr.predict(X_test) 
print("mean square error:%.1f"%mean_squared_error(y_test,y_pred))
print('R² score:%.2f'%r2_score(y_test,y_pred))



