from sklearn.preprocessing import LabelEncoder ,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 

my_data={
    "Age": [25, 30,pd.NA, 40],
    "Salary": [50000, pd.NA , 70000, 80000],
    "Gender":["Male", "Female", "Female", "Male"],
    "purchased":["Yes", "No", "Yes", "No"]

}
df=pd.DataFrame(my_data)
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Salary'].fillna(df['Salary'].mean(),inplace=True)
print(df)

le = LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['purchased']=df['purchased'].map({'Yes':1,'No':0})
print(df)

scaler=StandardScaler()
df[['Age','Salary']]=scaler.fit_transform(df[['Age','Salary']])
print(df)

X=df[['Age','Salary','Gender']]
y=df['purchased']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:\n",X_train)
print("X_test:\n",X_test)
print("y_train:\n",y_train)
print("y_test:\n",y_test)

