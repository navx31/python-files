import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('Exam_score.csv')
df_dup=df.copy()
df_dup.drop(['student_id',],axis=1,inplace=True)

df_dup = pd.get_dummies(df_dup, columns = ['age','gender','internet_access', 'study_method', 'sleep_quality', 'facility_rating', 'exam_difficulty', 'course'],dtype=int)   
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

model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

print("\n Model Performance ")
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R-squared:",r2)
print("Mean Absolute Error:",mae)

# user input for prediction
study_hours=float(input("Enter the number of study hours: "))
class_attendance=float(input("Enter the class attendance percentage: "))
sleep_hours=float(input("Enter the number of sleep hours: "))
internet_access=float(input("Do you have internet access? (1 for yes, 0 for no): "))
sleep_quality=float(input("Rate your sleep quality (1-5): "))
study_method=float(input("Choose your study method (1 for group, 0 for solo): "))
facility_rating=float(input("Rate your study facility (1-5): "))
exam_difficulty=float(input("Rate the exam difficulty (1-5): "))
course=float(input("Choose your course (1 for Math, 0 for English): "))

# build a raw single-row DataFrame from inputs (use numeric values for ordinal cats)
user_raw = pd.DataFrame([{
    'study_hours': study_hours,
    'class_attendance': class_attendance,
    'sleep_hours': sleep_hours,
    'internet_access': int(internet_access),
    'sleep_quality': int(sleep_quality),
    'study_method': int(study_method),
    'facility_rating': int(facility_rating),
    'exam_difficulty': int(exam_difficulty),
    'course': int(course),
    # provide placeholders for columns not asked interactively
    'age': 'unknown',
    'gender': 'unknown'
}])

# one-hot encode the same categorical columns used during training
cat_cols = ['age','gender','internet_access','study_method','sleep_quality','facility_rating','exam_difficulty','course']
user_ohe = pd.get_dummies(user_raw, columns=cat_cols, dtype=int)

# align to training features and fill missing dummy cols with 0
user_ohe = user_ohe.reindex(columns=X.columns, fill_value=0)

# scale and predict
user_scaled = scaler.transform(user_ohe)
predicted_score = model.predict(user_scaled)[0]
print(f"The predicted exam score is: {predicted_score:.2f}")
