import pandas as pd
import numpy as np 
import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the iris dataset
df=pd.read_csv("iris.csv")

# Preprocess the data
X = df.drop("species", axis=1)
y = df["species"]

# Encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# Split the data
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
per=Perceptron(max_iter=1000, random_state=42)
per.fit(X_train, y_train_cat)
y_pred = per.predict(X_test)
accuracy_score(y_test_cat, y_pred)
print(classification_report(y_test_cat, y_pred))
print(confusion_matrix(y_test_cat, y_pred))

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, to_categorical(y_train_cat), epochs=20, validation_split=0.1, verbose=1)
loss, accuracy = model.evaluate(X_test, to_categorical(y_test_cat))
print(f"Test Accuracy: {accuracy:.3f}")