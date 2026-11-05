from datetime import date
from pathlib import Path

import pandas as pd
from joblib import load


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model_v2.joblib"


def ask_integer(prompt, minimum, maximum=None):
    """Ask until a valid integer within the supported limits is entered."""
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum or (maximum is not None and value > maximum):
            if maximum is None:
                print(f"Please enter a value of at least {minimum}.")
            else:
                print(f"Please enter a value between {minimum} and {maximum}.")
            continue
        return value


def normalized(text):
    """Normalize spaces and capitalization for category matching."""
    return " ".join(text.split()).casefold()


def match_category(value, choices):
    """Return the model's original capitalization if a category is recognized."""
    by_normalized_name = {normalized(str(option)): str(option) for option in choices}
    return by_normalized_name.get(normalized(value))


def ask_category(prompt, choices):
    """Require a known value; match it case-insensitively."""
    options = [str(item) for item in choices]
    print(f"Options: {', '.join(options)}")
    while True:
        answer = input(f"{prompt}: ").strip()
        canonical = match_category(answer, options)
        if canonical is not None:
            return canonical
        print("Unknown value. Please choose one of the listed options.")


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Saved model not found: {MODEL_PATH}\n"
            "Make sure models/car_price_model_v2.joblib is present."
        )

    # Only load models created by you or from a source you fully trust.
    model = load(MODEL_PATH)
    encoder = model.named_steps["preprocessing"].named_transformers_["cat"]
    category_columns = [
        "brand", "fuel", "transmission", "seller_type", "owner", "model_name"
    ]
    categories = dict(zip(category_columns, encoder.categories_))

    print("\nUsed Car Price Prediction (INR)")
    while True:
        car_name = " ".join(input("Car name (e.g. Hyundai Verna 1.6 SX): ").split())
        if len(car_name.split()) >= 2:
            break
        print("Please enter a car name with at least two words.")

    year = ask_integer("Manufacturing year", 1886, date.today().year + 1)
    km_driven = ask_integer("Kilometers driven", 0)
    fuel = ask_category("Fuel type", categories["fuel"])
    transmission = ask_category("Transmission", categories["transmission"])
    seller_type = ask_category("Seller type", categories["seller_type"])
    owner = ask_category("Owner type", categories["owner"])

    # Match the exact feature extraction used in 01_explore.py / the saved model.
    parts = car_name.split()
    brand = match_category(parts[0], categories["brand"]) or parts[0]
    raw_model_name = " ".join(parts[:2])
    model_name = (
        match_category(raw_model_name, categories["model_name"]) or raw_model_name
    )
    if brand not in categories["brand"]:
        print("Warning: This brand was not present in the training data.")
    if model_name not in categories["model_name"]:
        print("Warning: This model name was not present in the training data.")
    if year > 2020:
        print("Warning: This manufacturing year is newer than the training data.")

    new_car = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "brand": brand,
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
        "owner": owner,
        "model_name": model_name,
    }])

    predicted_price = model.predict(new_car)[0]
    print(f"\nPredicted Selling Price: {predicted_price:,.2f} INR")
    print("Note: This is a model estimate, not a current market appraisal.")


