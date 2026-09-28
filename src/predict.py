import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/delivery_time_model.pkl")

# Example new order
new_order = pd.DataFrame({
    "distance_km": [8.5],
    "prep_time_min": [25],
    "traffic_level": [2],
    "rain": [0]
})

# Predict delivery time
prediction = model.predict(new_order)

print(f"Predicted delivery time: {prediction[0]:.2f} minutes")