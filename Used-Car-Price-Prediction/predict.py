
import pandas as pd
from joblib import load
from pathlib import Path

# Load trained model
model_path = Path(__file__).resolve().parent / "models" / "car_price_model_v2.joblib"
model = load(model_path)

# Get car information
car_name = input("Car name: ").strip()
year = int(input("Manufacturing year: "))
km_driven = int(input("Kilometers driven: "))
fuel = input("Fuel type: ").strip()
transmission = input("Transmission: ").strip()
seller_type = input("Seller type: ").strip()
owner = input("Owner type: ").strip()

# Extract brand and model
name_parts = car_name.split()

if len(name_parts) < 2:
    raise ValueError("Please enter at least two words for the car name.")

brand = name_parts[0]
model_name = " ".join(name_parts[:2])

# Create input DataFrame
new_car = pd.DataFrame([{
    "year": year,
    "km_driven": km_driven,
    "brand": brand,
    "fuel": fuel,
    "transmission": transmission,
    "seller_type": seller_type,
    "owner": owner,
    "model_name": model_name
}])

# Predict selling price
predicted_price = model.predict(new_car)[0]

print(f"\nPredicted Selling Price: {predicted_price:,.2f} INR")
