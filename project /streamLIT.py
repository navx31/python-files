import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ------------------- Page config -------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 Real Estate Price Predictor")
st.markdown("Enter the details of the house below to get an estimated price.")
st.markdown("![House Image](https://media.istockphoto.com/id/856794670/photo/beautiful-luxury-home-exterior-with-green-grass-and-landscaped-yard.jpg?s=612x612&w=is&k=20&c=jeAj7VBk-fqACzrhzL-_OxlAI6oJ0DYVkz_ZZmlT83Q=)")

# ------------------- Load and prepare data (cached) -------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv("real_estate.csv")
    df_dup = df.copy()
    df_dup.drop(columns=["ID", "Garage_Size", "Location_Score", "Distance_to_Center"], inplace=True, axis=1)

    num_cols = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms', 'Num_Floors', 'Year_Built']
    mm = MinMaxScaler()

    X = df_dup.drop('Price', axis=1)
    y = df_dup['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit scaler on training numeric columns, transform both
    mm.fit(X_train[num_cols])
    X_train[num_cols] = mm.transform(X_train[num_cols])
    X_test[num_cols] = mm.transform(X_test[num_cols])

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mm, num_cols, X.columns.tolist(), mse, r2, mae

model, scaler, num_cols, feature_names, mse, r2, mae = load_and_train()

# ------------------- Sidebar with model performance -------------------
st.sidebar.header("📊 Model Performance (test set)")
st.sidebar.metric("R² Score", f"{r2:.3f}")
st.sidebar.metric("Mean Absolute Error", f"${mae:,.0f}")
st.sidebar.metric("Mean Squared Error", f"${mse:,.0f}")

# ------------------- User input form -------------------
with st.form("prediction_form"):
    st.subheader("House Features")

    col1, col2 = st.columns(2)

    with col1:
        square_feet = st.number_input("Square Feet", min_value=100, max_value=10000, value=2000, step=50)
        num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
        num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
        num_floors = st.number_input("Number of Floors", min_value=1, max_value=5, value=2, step=1)

    with col2:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005, step=1)
        has_garden = st.checkbox("Has Garden?")
        has_pool = st.checkbox("Has Pool?")

    submitted = st.form_submit_button("Predict Price")

# ------------------- Prediction -------------------
if submitted:
    # Convert checkbox boolean to 1/0
    garden_val = 1 if has_garden else 0
    pool_val = 1 if has_pool else 0

    # Create input dataframe
    user_row = pd.DataFrame([[square_feet, num_bedrooms, num_bathrooms, num_floors, year_built, garden_val, pool_val]],
                            columns=feature_names)

    # Scale numeric columns
    user_row[num_cols] = scaler.transform(user_row[num_cols])

    # Predict
    prediction = model.predict(user_row)[0]

    # Show result
    st.success(f"### 💰 Predicted Price: **${prediction:,.2f}**")
    st.balloons()  # just for fun