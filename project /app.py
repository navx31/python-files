import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template_string

# -------------------------------
# 1. Train the Model (runs once at startup)
# -------------------------------
print("Loading dataset and training model...")
df = pd.read_csv("real_estate.csv")
df_dup = df.copy()

# Drop unused columns
df_dup.drop(columns=["ID", "Garage_Size", "Location_Score", "Distance_to_Center"], inplace=True, axis=1)

num_cols = ['Square_Feet', 'Num_Bedrooms', 'Num_Bathrooms', 'Num_Floors', 'Year_Built']
mm = MinMaxScaler()

# Split data
X = df_dup.drop('Price', axis=1)
y = df_dup['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale numeric features
mm.fit(X_train[num_cols])
X_train[num_cols] = mm.transform(X_train[num_cols])
X_test[num_cols] = mm.transform(X_test[num_cols])

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate (printed to console)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Model training complete.")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# -------------------------------
# 2. Flask App with Advanced UI (Bootstrap 5)
# -------------------------------
app = Flask(__name__)

# HTML Template (embedded as string)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding: 20px;
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.95);
        }
        .card-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px 15px 0 0 !important;
            padding: 20px;
            text-align: center;
            border: none;
        }
        .card-header h2 {
            margin: 0;
            font-weight: 600;
        }
        .card-body {
            padding: 30px;
        }
        .form-label {
            font-weight: 500;
            color: #495057;
        }
        .form-control, .form-select {
            border-radius: 10px;
            border: 1px solid #ced4da;
            padding: 10px 15px;
            transition: all 0.3s;
        }
        .form-control:focus, .form-select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102,126,234,0.25);
        }
        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            font-size: 1.1rem;
            color: white;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            background: linear-gradient(135deg, #5a6fd8 0%, #6a3f9c 100%);
        }
        .result-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 25px;
            text-align: center;
            border-left: 5px solid #667eea;
        }
        .result-price {
            font-size: 2rem;
            font-weight: 700;
            color: #28a745;
        }
        .input-group-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px 0 0 10px;
        }
        .info-icon {
            cursor: pointer;
            color: #6c757d;
        }
        .info-icon:hover {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="card">
                    <div class="card-header">
                        <h2>🏠 House Price Predictor</h2>
                        <p class="mb-0">Enter your property details below</p>
                    </div>
                    <div class="card-body">
                        <form method="POST" action="/predict">
                            <div class="row g-3">
                                <!-- Square Feet -->
                                <div class="col-md-6">
                                    <label class="form-label">Square Feet</label>
                                    <input type="number" step="any" class="form-control" name="square_feet" placeholder="e.g., 2500" min="0" required>
                                </div>
                                <!-- Bedrooms -->
                                <div class="col-md-6">
                                    <label class="form-label">Bedrooms</label>
                                    <input type="number" step="any" class="form-control" name="num_bedrooms" placeholder="e.g., 3" min="0" required>
                                </div>
                                <!-- Bathrooms -->
                                <div class="col-md-6">
                                    <label class="form-label">Bathrooms</label>
                                    <input type="number" step="any" class="form-control" name="num_bathrooms" placeholder="e.g., 2" min="0" required>
                                </div>
                                <!-- Year Built -->
                                <div class="col-md-6">
                                    <label class="form-label">Year Built</label>
                                    <input type="number" step="any" class="form-control" name="year_built" placeholder="e.g., 2005" min="1800" max="2026" required>
                                </div>
                                <!-- Floors -->
                                <div class="col-md-6">
                                    <label class="form-label">Floors</label>
                                    <input type="number" step="any" class="form-control" name="num_floors" placeholder="e.g., 2" min="1" required>
                                </div>
                                <!-- Garden -->
                                <div class="col-md-6">
                                    <label class="form-label">Has Garden?</label>
                                    <select class="form-select" name="has_garden" required>
                                        <option value="" disabled selected>Choose...</option>
                                        <option value="1">Yes</option>
                                        <option value="0">No</option>
                                    </select>
                                </div>
                                <!-- Pool -->
                                <div class="col-md-6">
                                    <label class="form-label">Has Pool?</label>
                                    <select class="form-select" name="has_pool" required>
                                        <option value="" disabled selected>Choose...</option>
                                        <option value="1">Yes</option>
                                        <option value="0">No</option>
                                    </select>
                                </div>
                                <!-- Submit Button -->
                                <div class="col-12 mt-4">
                                    <button type="submit" class="btn-predict">Predict Price</button>
                                </div>
                            </div>
                        </form>

                        {% if prediction_text %}
                        <div class="result-card">
                            <h5 class="mb-3">Your Estimated House Price</h5>
                            <div class="result-price">{{ prediction_text }}</div>
                            <small class="text-muted">* based on linear regression model</small>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    <!-- Bootstrap JS (optional, for some components) -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None)

@app.route('/predict', methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    # read form (HTML uses name="has_garden" and "has_pool")
    square_feet = float(request.form['square_feet'])
    num_bedrooms = float(request.form['num_bedrooms'])
    num_bathrooms = float(request.form['num_bathrooms'])
    year_built = float(request.form['year_built'])
    num_floors = float(request.form['num_floors'])
    has_garden = float(request.form['has_garden'])
    has_pool = float(request.form['has_pool'])

    # build input aligned to training feature names (X.columns)
    input_dict = {c: 0 for c in X.columns}
    # populate numeric features (num_cols order must match training)
    for c, v in zip(num_cols, [square_feet, num_bedrooms, num_bathrooms, num_floors, year_built]):
        input_dict[c] = v
    # map garden/pool to the exact column names used at training (case-sensitive)
    for c in X.columns:
        if c.lower() == 'has_garden':
            input_dict[c] = has_garden
        if c.lower() == 'has_pool':
            input_dict[c] = has_pool

    input_data = pd.DataFrame([input_dict], columns=X.columns)

    # Scale numeric columns and predict
    input_data[num_cols] = mm.transform(input_data[num_cols])
    prediction = model.predict(input_data)[0]
    formatted_price = f"${prediction:,.2f}"

    return render_template_string(HTML_TEMPLATE, prediction_text=formatted_price)


if __name__ == '__main__':
    app.run(debug=True)