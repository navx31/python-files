import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Exam_score.csv')
df_dup=df.copy()
df_dup.drop(['student_id','age'],axis=1,inplace=True)


df_dup = pd.get_dummies(df_dup, columns=['gender','internet_access','sleep_quality','study_method','facility_rating','exam_difficulty','course'], dtype=int)    
mm=MinMaxScaler()
df_dup['study_hours']=mm.fit_transform(df_dup['study_hours'].values.reshape(-1,1))
df_dup['class_attendance']=mm.fit_transform(df_dup['class_attendance'].values.reshape(-1,1))
df_dup['sleep_hours']=mm.fit_transform(df_dup['sleep_hours'].values.reshape(-1,1))

X=df_dup.drop('exam_score',axis=1)
y=df_dup['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsRegressor(n_neighbors=17)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)   

print("\n--- Model Performance ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")  

print("\n--- Actual vs Predicted ---")
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.tail())

