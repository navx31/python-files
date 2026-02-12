import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df= pd.read_csv('real_estate.csv')
df_dup=df.copy()
df_dup.drop(['ID','Garage_Size','Location_Score','Distance_to_Center'],axis=1,inplace=True)

mm=MinMaxScaler() 

df_dup['Square_Feet']=mm.fit_transform(df_dup['Square_Feet'].values.reshape(-1,1)) 
df_dup['Num_Bedrooms']=mm.fit_transform(df_dup['Num_Bedrooms'].values.reshape(-1,1)) 
df_dup['Num_Bathrooms']=mm.fit_transform(df_dup['Num_Bathrooms'].values.reshape(-1,1)) 
df_dup['Num_Floors']=mm.fit_transform(df_dup['Num_Floors'].values.reshape(-1,1)) 
df_dup['Year_Built']=mm.fit_transform(df_dup['Year_Built'].values.reshape(-1,1)) 
df_dup['Has_Garden']=mm.fit_transform(df_dup['Has_Garden'].values.reshape(-1,1)) 
df_dup['Has_Pool']=mm.fit_transform(df_dup['Has_Pool'].values.reshape(-1,1)) 

X=df_dup.drop('Price',axis=1)
y=df_dup['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# 4. FEATURE SCALING (optional but recommended for linear regression)
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 5. TRAIN LINEAR REGRESSION MODEL
# ------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ------------------------------
# 6. EVALUATE THE MODEL
# ------------------------------
y_pred = model.predict(X_test_scaled)
print("\n--- Model Performance ---")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# ------------------------------
# 7. PREDICT ON NEW USER INPUT
# ------------------------------
print("\n--- Predict House Price ---")
print("Enter the following details:")

def get_user_input():
    # Get numeric inputs with validation
    while True:
        try:
            sqft = float(input("Square feet: "))
            beds = int(input("Number of bedrooms: "))
            baths = int(input("Number of bathrooms: "))
            floors = int(input("Number of floors: "))
            garden = input("Has garden? (yes/no): ").strip().lower()
            pool = input("Has pool? (yes/no): ").strip().lower()
            year = int(input("Year built: "))
            break
        except ValueError:
            print("Invalid input. Please enter numeric values for numeric fields and yes/no for binary questions.")
    
    # Encode binary inputs
    garden_enc = 1 if garden == 'yes' else 0
    pool_enc = 1 if pool == 'yes' else 0
    
    # Create DataFrame with same column order as training
    user_data = pd.DataFrame([[sqft, beds, baths, floors, garden_enc, pool_enc, year]],
                             columns=X.columns)
    return user_data

user_df = get_user_input()
user_scaled = scaler.transform(user_df)
predicted_price = model.predict(user_scaled)[0]
print(f"\n💰 Predicted House Price: ${predicted_price:,.2f}")

# ------------------------------
# (Optional) Display feature importance
# ------------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\n--- Model Coefficients (standardized scale) ---")
print(coefficients.sort_values('Coefficient', key=abs, ascending=False).to_string(index=False))