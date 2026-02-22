import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('Exam_score.csv')
df_dup=df.copy()
df_dup.drop(['student_id','age'],axis=1,inplace=True)


df_dup = pd.get_dummies(df_dup, columns=['gender','internet_access','sleep_quality','study_method','facility_rating','exam_difficulty','course'], dtype=int)
mm=MinMaxScaler()
df_dup['study_hours']=mm.fit_transform(df_dup['study_hours'].values.reshape(-1,1))
df_dup['class_attendance']=mm.fit_transform(df_dup['class_attendance'].values.reshape(-1,1))
df_dup['sleep_hours']=mm.fit_transform(df_dup['sleep_hours'].values.reshape(-1,1))

X=df_dup.drop('exam_score',axis=1)
y=(df_dup['exam_score'] >= 50).astype(int)   # 1 = pass (>=50), 0 = fail
print("Class counts:", y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    

model=LogisticRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled) 

print("\n Model Performance ")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\n Classification Report ")
print(classification_report(y_test, y_pred))
print("\n Confusion Matrix ")
print(confusion_matrix(y_test, y_pred))

